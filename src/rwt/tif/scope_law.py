"""SCOPE-Law: sequence-conditioned portfolios of evolving transition laws.

This CPU-oriented research prototype keeps structural assignments soft during
training and carries complete structural/evolution hypotheses through online
inference. It is an approximate particle method, not an exact Bayesian filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import torch
from scipy.special import logsumexp

from .scope_support import Batch, hazard

SCOPE_PROGRAMS = tuple(product(range(3), repeat=2))


class EffectScriptDecoder(torch.nn.Module):
    """Shared transition core composed with an ordered pair of learned effects."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.effects = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(state_dim + action_dim, 48),
                                torch.nn.Tanh(), torch.nn.Linear(48, action_dim))
            for _ in range(3))
        self.transition_core = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, state_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("delta_mean", torch.zeros(state_dim))
        self.register_buffer("delta_scale", torch.ones(state_dim))

    def set_normalization(self, states: np.ndarray, deltas: np.ndarray) -> None:
        with torch.no_grad():
            self.state_mean.copy_(torch.as_tensor(states.mean(0), dtype=torch.float32))
            self.state_scale.copy_(torch.as_tensor(np.maximum(states.std(0), 0.1), dtype=torch.float32))
            self.delta_mean.copy_(torch.as_tensor(deltas.mean(0), dtype=torch.float32))
            self.delta_scale.copy_(torch.as_tensor(np.maximum(deltas.std(0), 0.01), dtype=torch.float32))

    def forward(self, states: torch.Tensor, actions: torch.Tensor, code: int,
                gains: torch.Tensor | None = None) -> torch.Tensor:
        normalized = (states - self.state_mean) / self.state_scale
        value = actions
        for effect_id in SCOPE_PROGRAMS[code]:
            effect = self.effects[effect_id](torch.cat((normalized, value), dim=-1))
            value = torch.tanh(value + 0.35 * torch.tanh(effect))
        if gains is not None:
            value = value * gains
        delta = self.transition_core(torch.cat((normalized, value), dim=-1))
        return states + self.delta_mean + self.delta_scale * delta


class AnchoredEffectDecoder(EffectScriptDecoder):
    """Action-identifiable skip plus ordered, zero-action-anchored effects.

    The skip is fitted jointly on public states and actions, so state/action
    correlation is accounted for. Neural residuals retain nonlinear capacity.
    This is an inductive bias, not an identifiability guarantee for latent codes.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        self.register_buffer("linear_skip", torch.zeros(state_dim + action_dim + 1, state_dim))
        self.register_buffer("likelihood_scale", torch.ones(state_dim))
        for effect in self.effects:
            torch.nn.init.zeros_(effect[-1].weight)
            torch.nn.init.zeros_(effect[-1].bias)
        torch.nn.init.zeros_(self.transition_core[-1].weight)
        torch.nn.init.zeros_(self.transition_core[-1].bias)

    def initialize_skip(self, states: np.ndarray, actions: np.ndarray,
                        targets: np.ndarray) -> None:
        normalized = (states - self.state_mean.numpy()) / self.state_scale.numpy()
        design = np.column_stack((normalized, actions, np.ones(len(states))))
        response = (targets - states - self.delta_mean.numpy()) / self.delta_scale.numpy()
        penalty = np.eye(design.shape[1]) * 0.5
        penalty[-1, -1] = 1e-6
        fitted = np.linalg.solve(design.T @ design + penalty, design.T @ response)
        self.linear_skip.copy_(torch.as_tensor(fitted, dtype=torch.float32))

    def forward(self, states: torch.Tensor, actions: torch.Tensor, code: int,
                gains: torch.Tensor | None = None) -> torch.Tensor:
        normalized = (states - self.state_mean) / self.state_scale
        zero = torch.zeros_like(actions)
        value = actions
        for effect_id in SCOPE_PROGRAMS[code]:
            effect = self.effects[effect_id]
            residual = effect(torch.cat((normalized, value), -1)) - effect(
                torch.cat((normalized, zero), -1))
            value = value + 0.35 * torch.tanh(residual)
        if gains is not None:
            value = value * gains
        design = torch.cat((normalized, value, torch.ones_like(value[..., :1])), -1)
        delta = design @ self.linear_skip + self.transition_core(torch.cat((normalized, value), -1))
        return states + self.delta_mean + self.delta_scale * delta


def _arrays(sequences: Sequence[Batch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("at least one public sequence is required")
    states = np.concatenate([item[0] for item in sequences])
    actions = np.concatenate([item[1] for item in sequences])
    targets = np.concatenate([item[2] for item in sequences])
    if states.ndim != 2 or actions.ndim != 2 or targets.shape != states.shape:
        raise ValueError("expected matching public state, action and next-state arrays")
    if len(states) < 16:
        raise ValueError("at least 16 public transitions required")
    return states, actions, targets


def _transition_matrix() -> np.ndarray:
    """Edit-aware prior over destinations, conditional on a regime switch."""
    count = len(SCOPE_PROGRAMS)
    result = np.empty((count, count), dtype=float)
    for i, left in enumerate(SCOPE_PROGRAMS):
        for j, right in enumerate(SCOPE_PROGRAMS):
            distance = sum(a != b for a, b in zip(left, right))
            # Staying is represented by the semi-Markov duration hazard. A
            # diagonal entry here would count persistence a second time and
            # distort both duration inference and the online switch rate.
            result[i, j] = 0.0 if i == j else np.exp(-0.7 * distance)
        result[i] /= result[i].sum()
    return result


def _forward_backward(emission: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Posterior code probabilities for one sequence of emission rows."""
    n, k = emission.shape
    if n == 0:
        return np.empty((0, k))
    log_transition = np.log(np.maximum(transition, 1e-300))
    # Emissions are mean standardized squared errors; temperature avoids
    # treating a small neural-model error difference as certain evidence.
    log_emission = -np.clip(emission, 0.0, 1e4) / 0.18
    forward = np.empty((n, k)); backward = np.zeros((n, k))
    forward[0] = -np.log(k) + log_emission[0]
    for t in range(1, n):
        forward[t] = log_emission[t] + logsumexp(
            forward[t - 1][:, None] + log_transition, axis=0)
    for t in range(n - 2, -1, -1):
        backward[t] = logsumexp(log_transition + log_emission[t + 1][None, :]
                                 + backward[t + 1][None, :], axis=1)
    log_posterior = forward + backward
    log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
    return np.exp(log_posterior)


def fit_effect_script(sequences: Sequence[Batch], *, seed: int = 0,
                      epochs: int = 5, window: int = 8,
                      anchored: bool = False, warmup_epochs: int = 2,
                      m_step_repeats: int = 1, validation_sequences=None,
                      validation_code: int | None = None
                      ) -> tuple[EffectScriptDecoder, np.ndarray, np.ndarray]:
    """Fit shared effects with a soft segmental E/M loop, never hard code labels."""
    if epochs < 1 or window < 2:
        raise ValueError("positive epochs and window >= 2 required")
    states, actions, targets = _arrays(sequences)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model_type = AnchoredEffectDecoder if anchored else EffectScriptDecoder
    model = model_type(states.shape[1], actions.shape[1])
    model.set_normalization(states, targets - states)
    if anchored:
        model.initialize_skip(states, actions, targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    x = torch.as_tensor(states, dtype=torch.float32)
    a = torch.as_tensor(actions, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    best_state, best_loss = None, float("inf")
    fit_trace = []
    if validation_sequences is not None:
        vx, va, vy = _arrays(validation_sequences)
        vx, va, vy = [torch.as_tensor(value, dtype=torch.float32) for value in (vx, va, vy)]

    def checkpoint(stage):
        nonlocal best_state, best_loss
        if validation_sequences is None:
            return
        with torch.no_grad():
            prediction = (torch.stack([model(vx, va, k) for k in range(9)]).mean(0)
                          if validation_code is None else model(vx, va, validation_code))
            score = float((((prediction-vy)/model.delta_scale)**2).mean())
        fit_trace.append({"stage": stage, "validation_nmse": score})
        if score < best_loss:
            best_loss = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    checkpoint("initial")
    windows: list[np.ndarray] = []
    sequence_windows: list[list[int]] = []
    offset = 0
    for state, _, _ in sequences:
        local = []
        for start in range(0, len(state), window):
            local.append(len(windows))
            windows.append(np.arange(offset + start, offset + min(start + window, len(state))))
        sequence_windows.append(local)
        offset += len(state)
    if not windows:
        raise ValueError("no transition windows")
    transition = _transition_matrix()
    # Warm-start the shared dynamics before asking the latent program model to
    # explain residual structure. This breaks the symmetric all-code initial
    # posterior that otherwise leaves both the script and duration laws flat.
    for warmup in range(warmup_epochs):
        for group in np.array_split(rng.permutation(len(windows)),
                                    max(1, int(np.ceil(len(windows) / 12)))):
            indices = np.concatenate([windows[int(wi)] for wi in group])
            prediction = model(x[indices], a[indices], 0)
            loss = (((prediction - y[indices]) / model.delta_scale) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        checkpoint(f"warmup_{warmup+1}")
    with torch.no_grad():
        pooled = model(x, a, 0).cpu().numpy()
    residual = (targets - pooled) / model.delta_scale.cpu().numpy()
    signatures = []
    for indices in windows:
        local = residual[indices]
        signatures.append(np.concatenate((local.mean(axis=0),
                                          (local.T @ actions[indices] / len(indices)).ravel())))
    signatures = np.stack(signatures)
    signatures /= np.maximum(signatures.std(axis=0), 0.1)
    centers = [signatures[int(rng.integers(len(signatures)))]]
    for _ in range(1, min(len(SCOPE_PROGRAMS), len(windows))):
        distance = np.min(np.stack([np.sum((signatures - center) ** 2, axis=1)
                                    for center in centers]), axis=0)
        centers.append(signatures[int(rng.choice(len(signatures),
                                                  p=distance / distance.sum()))]
                       if distance.sum() > 0 else signatures[int(rng.integers(len(signatures)))])
    centers = np.stack(centers)
    for _ in range(8):
        assignments = np.argmin(np.sum((signatures[:, None] - centers[None]) ** 2, axis=2),
                                axis=1)
        for code in range(len(centers)):
            if np.any(assignments == code):
                centers[code] = signatures[assignments == code].mean(axis=0)
    responsibilities = np.zeros((len(windows), len(SCOPE_PROGRAMS)), dtype=float)
    responsibilities[np.arange(len(windows)), assignments] = 1.0
    evolution_laws = _default_evolution_laws()
    evolution_prior = np.full(len(evolution_laws), 1 / len(evolution_laws), dtype=float)

    def expectation() -> np.ndarray:
        with torch.no_grad():
            predictions = [model(x, a, code).cpu().numpy() for code in range(len(SCOPE_PROGRAMS))]
        prediction = np.stack(predictions)
        scale = model.delta_scale.cpu().numpy()
        emissions = np.empty_like(responsibilities)
        for wi, indices in enumerate(windows):
            for code in range(len(SCOPE_PROGRAMS)):
                residual = (prediction[code, indices] - targets[indices]) / scale
                # The dynamic program consumes log likelihoods, not costs.
                # Keeping the sign negative is essential: otherwise the E step
                # rewards scripts with larger prediction errors.
                emissions[wi, code] = -0.5 * np.mean(residual ** 2)
        posterior = np.zeros_like(responsibilities)
        switch_counts = np.full((len(SCOPE_PROGRAMS), len(SCOPE_PROGRAMS)), 0.15)
        np.fill_diagonal(switch_counts, 0.0)
        posterior_evolution = np.zeros(len(evolution_laws), dtype=float)
        for indices in sequence_windows:
            evidence = np.empty(len(evolution_laws), dtype=float)
            path_posteriors = []
            path_switch_counts = []
            for evolution_id, law in enumerate(evolution_laws):
                path, log_evidence, switches = _semi_markov_forward_backward(
                    emissions[indices], transition, law, window)
                evidence[evolution_id] = log_evidence
                path_posteriors.append(path)
                path_switch_counts.append(switches)
            log_weights = np.log(np.maximum(evolution_prior, 1e-300)) + evidence
            weights = np.exp(log_weights - logsumexp(log_weights))
            posterior_evolution += weights / len(sequence_windows)
            for weight, path, switches in zip(weights, path_posteriors, path_switch_counts):
                posterior[indices] += weight * path
                switch_counts += weight * switches
        np.fill_diagonal(switch_counts, 0.0)
        learned_transition = switch_counts / np.maximum(
            switch_counts.sum(axis=1, keepdims=True), 1e-12)
        return posterior, learned_transition, posterior_evolution

    # Generalized EM: posterior paths stay distributions. The M step weights
    # each window by its full sequence posterior instead of emitting a pseudo-label.
    for epoch in range(epochs):
        if epoch:
            responsibilities, transition, evolution_prior = expectation()
        order = np.concatenate([rng.permutation(len(windows)) for _ in range(m_step_repeats)])
        for group in np.array_split(order, max(1, int(np.ceil(len(order) / 12)))):
            terms = []
            for wi in group:
                indices = windows[int(wi)]
                weights = responsibilities[int(wi)]
                for code in np.flatnonzero(weights > 1e-4):
                    pred = model(x[indices], a[indices], int(code))
                    mse = (((pred - y[indices]) / model.delta_scale) ** 2).mean()
                    terms.append(float(weights[code]) * mse)
            loss = torch.stack(terms).sum() / max(1, len(group))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        checkpoint(f"em_{epoch+1}")
    if best_state is not None:
        model.load_state_dict(best_state)
    responsibilities, transition, evolution_prior = expectation()
    model.fit_trace = fit_trace
    model.eval()
    return model, transition, evolution_prior


@dataclass(frozen=True)
class EvolutionLaw:
    scale: float
    shape: float
    drift: float
    jump_rate: float
    jump_std: float


def _default_evolution_laws() -> tuple[EvolutionLaw, ...]:
    return tuple(EvolutionLaw(scale, shape, drift, jump, jump_std)
                 for scale in (8., 18., 42., 96.)
                 for shape in (1., 2.)
                 for drift, jump, jump_std in ((0.001, 0.01, 0.035),
                                               (0.004, 0.05, 0.08),
                                               (0.0025, 0.025, 0.06),
                                               (0.0025, 0.05, 0.035)))


def _semi_markov_forward_backward(emission: np.ndarray, transition: np.ndarray,
                                  law: EvolutionLaw, block: int
                                  ) -> tuple[np.ndarray, float, np.ndarray]:
    """Age-augmented forward-backward with right-censored terminal survival."""
    length, codes = emission.shape
    if length == 0:
        return np.empty((0, codes)), 0.0, np.zeros((codes, codes))
    ages = length
    age_grid = np.arange(ages)
    # One latent state labels an entire observation window. Its transition
    # probability must therefore cover all `block` simulator steps. Using the
    # one-step hazard at the window boundary systematically overstates dwell
    # time and distorts the shape likelihood.
    switch = np.asarray([
        -np.expm1(-(((age + 1) * block / law.scale) ** law.shape
                    - (age * block / law.scale) ** law.shape))
        for age in age_grid
    ])
    log_switch = np.log(np.maximum(switch, 1e-300))
    log_stay = np.log(np.maximum(1.0 - switch, 1e-300))
    log_transition = np.log(np.maximum(transition, 1e-300))
    log_transition[np.asarray(transition) <= 0] = -np.inf
    alpha = np.full((length, codes, ages), -np.inf)
    beta = np.zeros((length, codes, ages), dtype=float)
    alpha[0, :, 0] = -np.log(codes) + emission[0]
    next_age = np.minimum(age_grid + 1, ages - 1)
    for t in range(1, length):
        prior = np.full((codes, ages), -np.inf)
        for age, destination in enumerate(next_age):
            prior[:, destination] = np.logaddexp(
                prior[:, destination], alpha[t - 1, :, age] + log_stay[age])
        switch_mass = logsumexp(alpha[t - 1] + log_switch[None, :], axis=1)
        prior[:, 0] = np.logaddexp(prior[:, 0],
                                   logsumexp(switch_mass[:, None] + log_transition, axis=0))
        alpha[t] = prior + emission[t, :, None]
    log_evidence = float(logsumexp(alpha[-1]))
    for t in range(length - 2, -1, -1):
        for age, destination in enumerate(next_age):
            stay_value = (log_stay[age] + emission[t + 1]
                          + beta[t + 1, :, destination])
            switch_value = (log_switch[age] + logsumexp(
                log_transition + (emission[t + 1] + beta[t + 1, :, 0])[None, :], axis=1))
            beta[t, :, age] = np.logaddexp(stay_value, switch_value)
    posterior_log = alpha + beta - log_evidence
    posterior = np.exp(logsumexp(posterior_log, axis=2))
    switch_counts = np.zeros((codes, codes), dtype=float)
    for t in range(length - 1):
        mass = (alpha[t, :, :, None] + log_switch[None, :, None]
                + log_transition[:, None, :]
                + (emission[t + 1] + beta[t + 1, :, 0])[None, None, :]
                - log_evidence)
        # Sum over durations and time for each ordered old/new code pair.
        switch_counts += np.exp(np.clip(logsumexp(mass, axis=1), -745, 0))
    np.fill_diagonal(switch_counts, 0.0)
    return posterior, log_evidence, switch_counts


@dataclass
class _LawParticle:
    code: int
    gains: np.ndarray
    age: int
    evolution: int
    weight: float


class ScopeLaw:
    """Joint particle belief over current law and its future evolution."""

    method_type = "scope_law_particle_portfolio"
    capabilities = ("current_kernel", "future_law_samples")

    def __init__(self, model: EffectScriptDecoder, transition: np.ndarray,
                 noise_std: float, *, seed: int = 0, particles: int = 192,
                 evolution_laws: Sequence[EvolutionLaw] | None = None,
                 likelihood_temperature: float = 1.0,
                 evolution_prior: np.ndarray | None = None,
                 kernel_noise_std: float | None = None):
        if particles < 16 or noise_std <= 0 or likelihood_temperature <= 0:
            raise ValueError("at least 16 particles and positive likelihood scales required")
        self.model = model.eval()
        self.transition = np.asarray(transition, dtype=float)
        self.noise_std = float(noise_std)
        self.kernel_noise_std = float(noise_std if kernel_noise_std is None else kernel_noise_std)
        if not np.isfinite(self.kernel_noise_std) or self.kernel_noise_std < 0:
            raise ValueError("kernel noise must be finite and nonnegative")
        self.rng = np.random.default_rng(seed)
        self.forecast_rng = np.random.default_rng(int(seed) + 81_001)
        self.evolution_laws = tuple(evolution_laws or _default_evolution_laws())
        if particles < max(16, len(self.evolution_laws)):
            raise ValueError("particle count must cover every evolution hypothesis")
        prior = (np.full(len(self.evolution_laws), 1 / len(self.evolution_laws))
                 if evolution_prior is None else np.asarray(evolution_prior, dtype=float))
        if prior.shape != (len(self.evolution_laws),) or np.any(prior < 0) or prior.sum() <= 0:
            raise ValueError("evolution prior must be a nonnegative weight per candidate law")
        self.evolution_prior = prior / prior.sum()
        # Cover every positive-mass evolution hypothesis instead of losing
        # roughly a third when n == number of hypotheses (occupancy sampling).
        active = np.flatnonzero(self.evolution_prior > 0)
        remaining = self.rng.choice(len(self.evolution_laws), size=particles - len(active),
                                    p=self.evolution_prior)
        evolution_ids = np.concatenate((active, remaining))
        counts = np.bincount(evolution_ids, minlength=len(self.evolution_laws))
        code_ids = np.resize(self.rng.permutation(len(SCOPE_PROGRAMS)), particles)
        self.rng.shuffle(code_ids)
        self.particles = [_LawParticle(int(code),
                                       np.ones(self.model.action_dim), 0, int(evolution),
                                       float(self.evolution_prior[evolution] / counts[evolution]))
                          for evolution, code in zip(evolution_ids, code_ids)]
        self.likelihood_temperature = float(likelihood_temperature)
        self._latest_time = 0
        self.filter_trace: list[dict[str, float]] = []

    def _candidate_branches(self, particle: _LawParticle, rng=None
                            ) -> list[tuple[_LawParticle, float]]:
        """Enumerate stay and structural-switch proposals for one parent law."""
        law = self.evolution_laws[particle.evolution]
        rng = self.rng if rng is None else rng
        switch_prob = hazard(particle.age, law.scale, law.shape)
        branches: list[tuple[_LawParticle, float]] = []
        action_dim = self.model.action_dim
        stay_gain = particle.gains + rng.normal(0.0, law.drift, action_dim)
        jumps = int(rng.poisson(law.jump_rate))
        if jumps:
            stay_gain += rng.normal(0.0, law.jump_std,
                                         (jumps, action_dim)).sum(axis=0)
        branches.append((_LawParticle(particle.code, np.clip(stay_gain, 0.55, 1.45),
                                      particle.age + 1, particle.evolution,
                                      particle.weight), 1.0 - switch_prob))
        switch_gain = np.clip(particle.gains + rng.normal(0.0, 0.025,
                                                               action_dim),
                              0.55, 1.45)
        for code, probability in enumerate(self.transition[particle.code]):
            if probability > 0 and code != particle.code:
                branches.append((_LawParticle(code, switch_gain.copy(), 0,
                                              particle.evolution, particle.weight),
                                 switch_prob * float(probability)))
        return branches

    def _sample_advance(self, particle: _LawParticle, rng=None) -> _LawParticle:
        rng = self.rng if rng is None else rng
        branches = self._candidate_branches(particle, rng)
        probabilities = np.asarray([probability for _, probability in branches], dtype=float)
        probabilities /= probabilities.sum()
        return branches[int(rng.choice(len(branches), p=probabilities))][0]

    def _resample_portfolio(self, candidates: list[_LawParticle],
                            weights: np.ndarray) -> None:
        """Systematic resampling within every evolution law; retain its mass."""
        n = len(self.particles)
        masses = np.bincount([p.evolution for p in candidates], weights=weights,
                             minlength=len(self.evolution_laws))
        masses /= masses.sum()
        ideal = masses * n
        # Zero-mass hypotheses need no particles. Giving them a minimum slot
        # and filling the resulting gaps from the global posterior duplicates
        # weights and breaks normalization after resampling.
        allocation = np.where(masses > 0, np.maximum(1, np.floor(ideal).astype(int)), 0)
        while allocation.sum() < n:
            scores = np.where(masses > 0, ideal - allocation, -np.inf)
            allocation[int(np.argmax(scores))] += 1
        while allocation.sum() > n:
            eligible = np.flatnonzero(allocation > 1)
            if not len(eligible):
                raise RuntimeError("cannot allocate particles across positive-mass hypotheses")
            scores = allocation[eligible] - ideal[eligible]
            allocation[eligible[int(np.argmax(scores))]] -= 1
        selected: list[_LawParticle] = []
        candidate_evolution = np.asarray([p.evolution for p in candidates])
        for evolution, count in enumerate(allocation):
            if count == 0 or masses[evolution] <= 0:
                continue
            indices = np.flatnonzero(candidate_evolution == evolution)
            conditional = weights[indices] / masses[evolution]
            conditional /= conditional.sum()
            positions = (self.rng.random() + np.arange(count)) / count
            chosen = indices[np.searchsorted(np.cumsum(conditional), positions, side="right")]
            for index in chosen:
                item = candidates[int(index)]
                selected.append(_LawParticle(item.code, item.gains.copy(), item.age,
                                             item.evolution, float(masses[evolution] / count)))
        if len(selected) != n:
            raise RuntimeError(f"resampler produced {len(selected)} particles, expected {n}")
        self.particles = selected

    def evolution_effective_count(self) -> float:
        masses = np.bincount([p.evolution for p in self.particles],
                             weights=[p.weight for p in self.particles],
                             minlength=len(self.evolution_laws))
        masses = masses / max(float(masses.sum()), 1e-300)
        return float(1.0 / np.sum(masses ** 2))

    def observe(self, fragment: Batch) -> None:
        states, actions, targets = (np.asarray(part) for part in fragment)
        if not (len(states) == len(actions) == len(targets)):
            raise ValueError("transition lengths differ")
        for state, action, target in zip(states, actions, targets):
            if self._latest_time:
                candidates = []
                log_prior = []
                for particle in self.particles:
                    for branch, probability in self._candidate_branches(particle):
                        if probability > 0:
                            candidates.append(branch)
                            log_prior.append(np.log(max(particle.weight, 1e-300))
                                             + np.log(probability))
            else:
                candidates = self.particles
                log_prior = np.log(np.maximum([p.weight for p in candidates], 1e-300))
            log_weights = np.asarray(log_prior, dtype=float)
            # Likelihood-informed branch enumeration prevents rare switch paths
            # from disappearing just because the bootstrap proposal missed them.
            for code in range(len(SCOPE_PROGRAMS)):
                where = [i for i, p in enumerate(candidates) if p.code == code]
                if not where:
                    continue
                gains = torch.as_tensor(np.stack([candidates[i].gains for i in where]),
                                        dtype=torch.float32)
                x = torch.as_tensor(np.repeat(state[None], len(where), axis=0), dtype=torch.float32)
                a = torch.as_tensor(np.repeat(action[None], len(where), axis=0), dtype=torch.float32)
                with torch.no_grad():
                    prediction = self.model(x, a, code, gains).cpu().numpy()
                scale = getattr(self.model, "likelihood_scale", self.model.delta_scale)
                delta_scale = np.maximum(scale.cpu().numpy(), self.noise_std)
                scaled_error = (prediction - target[None]) / delta_scale[None]
                errors = np.mean(scaled_error ** 2, axis=1)
                log_weights[where] += -0.5 * errors / self.likelihood_temperature
            log_weights -= logsumexp(log_weights)
            weights = np.exp(log_weights)
            self._resample_portfolio(candidates, weights)
            self._latest_time += 1
            code_mass = np.bincount([p.code for p in self.particles],
                                    weights=[p.weight for p in self.particles],
                                    minlength=len(SCOPE_PROGRAMS))
            code_mass /= max(float(code_mass.sum()), 1e-300)
            self.filter_trace.append({
                "timestep": float(self._latest_time - 1),
                "effective_codes": float(1.0 / np.sum(code_mass ** 2)),
                "effective_evolution_laws": self.evolution_effective_count(),
            })

    def _kernel(self, code: int, gains: np.ndarray,
                probes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            means = self.model(torch.as_tensor(probes[0], dtype=torch.float32),
                               torch.as_tensor(probes[1], dtype=torch.float32), code,
                               torch.as_tensor(np.repeat(gains[None], len(probes[0]), axis=0),
                                               dtype=torch.float32)).cpu().numpy()
        variance = np.full(means.size, self.kernel_noise_std ** 2)
        return np.concatenate((means.ravel(), variance))

    def predict_kernel(self, probes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        weights = np.asarray([p.weight for p in self.particles])
        weights /= weights.sum()
        return sum(weight * self._kernel(p.code, p.gains, probes)
                   for weight, p in zip(weights, self.particles))

    def forecast_laws(self, horizons: tuple[int, ...],
                      probes: tuple[np.ndarray, np.ndarray],
                      n_samples: int = 32) -> dict[int, np.ndarray]:
        if not horizons or min(horizons) < 1 or n_samples < 2:
            raise ValueError("positive horizons and at least two samples required")
        weights = np.asarray([p.weight for p in self.particles], dtype=float)
        weights /= weights.sum()
        sampled = self.forecast_rng.choice(len(self.particles), size=n_samples, p=weights)
        max_horizon = max(horizons)
        paths = [self.particles[int(i)] for i in sampled]
        output = {h: np.empty((n_samples, 2 * len(probes[0]) * self.model.state_dim))
                  for h in horizons}
        for step in range(1, max_horizon + 1):
            paths = [self._sample_advance(p, self.forecast_rng) for p in paths]
            if step in output:
                output[step] = np.stack([self._kernel(p.code, p.gains, probes) for p in paths])
        return output
