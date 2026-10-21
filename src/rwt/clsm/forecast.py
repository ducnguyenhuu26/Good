"""Pure future-law push-forward, independent from the online filter RNG."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .belief import LawComponent


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
        for step in range(max(horizons) + 1):
            if step in horizons:
                out[step] = torch.as_tensor(np.stack([np.r_[code, theta] for code, _, theta, _ in paths]), dtype=torch.float64)
            if step == max(horizons) or frozen:
                continue
            next_paths = []
            for code, age, theta, hypothesis in paths:
                h = float(self.evolution.hazard(torch.tensor([code]), torch.tensor([age]))[0].detach())
                if rng.random() < h:
                    probabilities = self.evolution.structural_log_transition()[code].detach().exp().cpu().numpy()
                    new_code = int(rng.choice(8, p=probabilities))
                    new_age = 0
                    event = "entry"
                else:
                    new_code, new_age = code, age + 1
                    event = "entry" if rng.random() < float(self.evolution.reset_prob.detach()) else "drift"
                active = np.array([(new_code >> j) & 1 for j in range(3)], dtype=bool)
                new_theta = np.zeros(3, dtype=np.float64)
                if active.any():
                    if event == "entry":
                        mean, cov = self.memory.prior_for_entry(new_code, hypothesis)
                        new_theta[active] = rng.multivariate_normal(mean.numpy()[active], cov.numpy()[np.ix_(active, active)])
                    else:
                        new_theta[active] = theta[active] + rng.normal(0, np.sqrt(self.evolution.q.detach().numpy()[active]))
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
                    signatures.append((mean - obs).flatten())
            result[h] = torch.stack(signatures)
        return result
