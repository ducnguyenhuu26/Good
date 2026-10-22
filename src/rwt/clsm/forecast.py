"""Pure future-law push-forward, independent from the online filter RNG."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .belief import LawComponent
from .inference_dp import AgeLawFilter


class LawForecaster:
    def __init__(self, model, evolution, memory, samples: int = 256):
        if samples < 2:
            raise ValueError("at least two forecast samples required")
        self.model, self.evolution, self.memory, self.samples = model, evolution, memory, samples

    def sample_laws(self, belief: list[LawComponent], horizons: list[int], rng: np.random.Generator,
                    frozen: bool = False) -> dict[int, Tensor]:
        if not horizons or min(horizons) < 0:
            raise ValueError("horizons must be nonnegative")
        weights = torch.stack([c.log_weight.exp() for c in belief]).cpu().numpy()
        weights /= weights.sum()
        indices = rng.choice(len(belief), size=self.samples, p=weights)
        paths = []
        for i in indices:
            item = belief[int(i)]
            mask = torch.tensor([(item.code >> j) & 1 for j in range(3)], dtype=torch.bool)
            mean = item.mean.cpu().numpy()
            cov = item.cov.cpu().numpy()
            theta = np.zeros(3, dtype=np.float64)
            active = mask.numpy()
            if active.any():
                theta[active] = rng.multivariate_normal(mean[active], cov[np.ix_(active, active)])
            paths.append((item.code, item.age, theta, item.scratch_or_reuse))
        out = {}
        initial_laws = torch.as_tensor(np.stack([np.r_[code, theta] for code, _, theta, _ in paths]), dtype=torch.float64)
        with torch.no_grad():
            max_age = max(age for _, age, _, _ in paths) + max(horizons) + 1
            code_grid = torch.arange(8, dtype=torch.long)[:, None].expand(8, max_age)
            age_grid = torch.arange(max_age, dtype=torch.long)[None, :].expand(8, max_age)
            hazards = self.evolution.hazard(code_grid, age_grid).cpu().numpy()
            transitions = self.evolution.structural_log_transition().exp().cpu().numpy()
            reset_probability = float(self.evolution.reset_prob)
            drift_std = np.sqrt(self.evolution.q.cpu().numpy())
        for step in range(max(horizons) + 1):
            if step in horizons:
                out[step] = torch.as_tensor(np.stack([np.r_[code, theta] for code, _, theta, _ in paths]), dtype=torch.float64)
            if step == max(horizons) or frozen:
                continue
            next_paths = []
            for code, age, theta, hypothesis in paths:
                h = hazards[code, age]
                if rng.random() < h:
                    new_code = int(rng.choice(8, p=transitions[code]))
                    new_age = 0
                    event = "entry"
                else:
                    new_code, new_age = code, age + 1
                    event = "entry" if rng.random() < reset_probability else "drift"
                active = np.array([(new_code >> j) & 1 for j in range(3)], dtype=bool)
                new_theta = np.zeros(3, dtype=np.float64)
                if active.any():
                    if event == "entry":
                        mean, cov = self.memory.prior_for_entry(new_code, hypothesis)
                        new_theta[active] = rng.multivariate_normal(mean.numpy()[active], cov.numpy()[np.ix_(active, active)])
                    else:
                        new_theta[active] = theta[active] + rng.normal(0, drift_std[active])
                next_paths.append((new_code, new_age, new_theta, hypothesis))
            paths = next_paths
        if frozen:
            for h in horizons:
                out[h] = initial_laws.clone()
        return out

    def sample_signatures(self, belief: list[LawComponent], probes: tuple[Tensor, Tensor],
                          horizons: list[int], rng: np.random.Generator, frozen: bool = False) -> dict[int, Tensor]:
        obs, action = probes
        if obs.ndim != 2 or action.ndim != 2 or len(obs) != len(action):
            raise ValueError("probe batch mismatch")
        laws = self.sample_laws(belief, horizons, rng, frozen=frozen)
        result = {}
        for h, values in laws.items():
            signatures = []
            with torch.no_grad():
                for row in values:
                    code = torch.full((len(obs),), int(row[0]), dtype=torch.long)
                    theta = row[1:].expand(len(obs), 3)
                    mean = self.model(obs, action, code, theta)[0]
                    signatures.append(((mean - obs) / self.model.delta_scale).flatten())
            result[h] = torch.stack(signatures)
        return result


class AgeLawForecaster:
    """Forecast from exact code-age mass and factored strength Gaussians."""

    def __init__(self, model, evolution, memory, samples: int = 256):
        if samples < 2:
            raise ValueError("at least two forecast samples required")
        self.model, self.evolution, self.memory, self.samples = model, evolution, memory, samples

    def sample_laws(self, belief: AgeLawFilter, horizons: list[int], rng: np.random.Generator,
                    frozen: bool = False) -> dict[int, Tensor]:
        if not horizons or min(horizons) < 0 or max(horizons) > belief.max_world_length - belief.time:
            raise ValueError("forecast horizons must fit the remaining finite world")
        probabilities = belief.probabilities.detach().cpu().numpy().reshape(-1)
        probabilities /= probabilities.sum()
        selected = rng.choice(len(probabilities), size=self.samples, p=probabilities)
        paths = []
        for index in selected:
            code, age = divmod(int(index), belief.log_code_age.shape[1])
            active = np.array([(code >> j) & 1 for j in range(3)], dtype=bool)
            mean = belief.xi_mean[code].detach().cpu().numpy()
            cov = belief.xi_cov[code].detach().cpu().numpy()
            xi = np.zeros(3, dtype=np.float64)
            if active.any():
                xi[active] = rng.multivariate_normal(mean[active], cov[np.ix_(active, active)])
            paths.append((code, age, xi))
        transition = self.evolution.structural_log_transition().detach().exp().cpu().numpy()
        drift_std = np.sqrt(self.evolution.q.detach().cpu().numpy())
        reset = np.array([float(self.evolution.reset_probability(code).detach()) for code in range(8)])
        result = {}
        for step in range(max(horizons) + 1):
            if step in horizons:
                result[step] = torch.as_tensor(np.stack([[code, *xi] for code, _, xi in paths]),
                                                dtype=torch.float64)
            if step == max(horizons) or frozen:
                continue
            advanced = []
            for code, age, xi in paths:
                h = float(self.evolution.hazard(torch.tensor([code]), torch.tensor([age]))[0].detach())
                if rng.random() < h:
                    target = int(rng.choice(8, p=transition[code]))
                    target_age = 0
                    entry = True
                else:
                    target = code
                    target_age = age + 1
                    entry = (target != 0 and rng.random() < reset[target])
                active = np.array([(target >> j) & 1 for j in range(3)], dtype=bool)
                new_xi = np.zeros(3, dtype=np.float64)
                if active.any():
                    if entry:
                        mean, covariance = self.memory.entry_prior(target)
                        mean, covariance = mean.detach().cpu().numpy(), covariance.detach().cpu().numpy()
                        new_xi[active] = rng.multivariate_normal(mean[active], covariance[np.ix_(active, active)])
                    else:
                        new_xi[active] = xi[active] + rng.normal(0., drift_std[active])
                advanced.append((target, target_age, new_xi))
            paths = advanced
        return result

    def sample_signatures(self, belief: AgeLawFilter, probes: tuple[Tensor, Tensor],
                          horizons: list[int], rng: np.random.Generator,
                          frozen: bool = False) -> dict[int, Tensor]:
        obs, action = probes
        if obs.ndim != 2 or action.ndim != 2 or len(obs) != len(action):
            raise ValueError("probe batch mismatch")
        laws = self.sample_laws(belief, horizons, rng, frozen=frozen)
        outputs = {}
        with torch.no_grad():
            for horizon, samples in laws.items():
                signatures = []
                for row in samples:
                    code = torch.full((len(obs),), int(row[0]), dtype=torch.long, device=obs.device)
                    xi = row[1:].to(device=obs.device, dtype=obs.dtype).expand(len(obs), 3)
                    mean = self.model(obs, action, code, xi)[0]
                    signatures.append(((mean - obs) / self.model.delta_scale).flatten())
                outputs[horizon] = torch.stack(signatures)
        return outputs
