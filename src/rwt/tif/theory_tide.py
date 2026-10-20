"""TheoryTide: a compact executable-program filter for changing dynamics.

This is an in-house research method, not a renamed paper baseline.  It learns
ordered actuator primitives from public transitions, a duration-sensitive
program transition model, and a weighted belief over programs and gains.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.stats import t as student_t
import torch
from torch import nn

from .scope_support import Batch, hazard


PROGRAMS = tuple(product(range(3), repeat=2))


class MechanismLoom(nn.Module):
    """Shared transition model with a learned, ordered action grammar."""

    def __init__(self, state_dim: int, action_dim: int, *, zero_action_anchor: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.zero_action_anchor = zero_action_anchor
        self.primitives = nn.ModuleList(
            nn.Linear(state_dim + action_dim, action_dim) for _ in range(3))
        self.dynamics = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.Tanh(),
                                      nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, state_dim))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("delta_mean", torch.zeros(state_dim))
        self.register_buffer("delta_scale", torch.ones(state_dim))

    def set_normalization(self, states: np.ndarray, deltas: np.ndarray) -> None:
        with torch.no_grad():
            self.state_mean.copy_(torch.as_tensor(states.mean(axis=0), dtype=torch.float32))
            self.state_scale.copy_(torch.as_tensor(np.maximum(states.std(axis=0), 0.1),
                                                   dtype=torch.float32))
            self.delta_mean.copy_(torch.as_tensor(deltas.mean(axis=0), dtype=torch.float32))
            self.delta_scale.copy_(torch.as_tensor(np.maximum(deltas.std(axis=0), 0.01),
                                                   dtype=torch.float32))

    def forward(self, states: torch.Tensor, actions: torch.Tensor,
                code: int, gains: torch.Tensor | None = None) -> torch.Tensor:
        standardized = (states - self.state_mean) / self.state_scale
        value = actions
        for primitive in PROGRAMS[code]:
            response = self.primitives[primitive](torch.cat((standardized, value), -1))
            if self.zero_action_anchor:
                zero_action = torch.zeros_like(value)
                baseline = self.primitives[primitive](
                    torch.cat((standardized, zero_action), -1))
                response = response - baseline
            change = torch.tanh(response)
            value = torch.tanh(value + 0.35 * change)
        if gains is not None:
            value = value * gains
        residual = self.dynamics(torch.cat((standardized, value), -1))
        return states + self.delta_mean + self.delta_scale * residual


def _arrays(sequences: Sequence[Batch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.concatenate([part[0] for part in sequences])
    actions = np.concatenate([part[1] for part in sequences])
    targets = np.concatenate([part[2] for part in sequences])
    if states.ndim != 2 or actions.ndim != 2 or targets.shape != states.shape:
        raise ValueError("expected matching public state, action, next-state arrays")
    if len(states) < 16:
        raise ValueError("at least 16 public transitions required")
    return states, actions, targets


def fit_mechanism_loom(sequences: Sequence[Batch], *, seed: int = 0,
                       epochs: int = 6, window: int = 8,
                       zero_action_anchor: bool = False) -> MechanismLoom:
    """Fit shared primitives on public windows with residual-based proposals."""
    if epochs < 1 or window < 2:
        raise ValueError("positive epochs and window >= 2 required")
    states, actions, targets = _arrays(sequences)
    torch.manual_seed(seed)
    model = MechanismLoom(states.shape[1], actions.shape[1],
                          zero_action_anchor=zero_action_anchor)
    model.set_normalization(states, targets - states)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    # Each window has one latent program proposal; no simulator labels enter.
    windows = []
    offset = 0
    for sequence in sequences:
        count = len(sequence[0])
        windows.extend(np.arange(offset + start, offset + min(start + window, count))
                       for start in range(0, count, window))
        offset += count
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(states, dtype=torch.float32)
    a = torch.as_tensor(actions, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    # Learn the common dynamics first.  Then cluster *public prediction
    # residuals*, not simulator theory labels, to break mixture collapse.
    for _ in range(2):
        order = rng.permutation(len(windows))
        for group in np.array_split(order, max(1, int(np.ceil(len(order) / 8)))):
            indices = np.concatenate([windows[int(i)] for i in group])
            predicted = model(x[indices], a[indices], 0)
            loss = (((predicted - y[indices]) / model.delta_scale) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        pooled = model(x, a, 0).numpy()
    residual = (targets - pooled) / model.delta_scale.numpy()
    signatures = []
    for indices in windows:
        local = residual[indices]
        action = actions[indices]
        signatures.append(np.concatenate((local.mean(0),
                                          (local.T @ action / len(indices)).ravel())))
    signatures = np.stack(signatures)
    signatures /= np.maximum(signatures.std(0), 0.1)
    n_groups = min(len(PROGRAMS), len(windows))
    centers = [signatures[int(rng.integers(len(signatures)))]]
    for _ in range(1, n_groups):
        distance = np.min(np.stack([np.sum((signatures - center) ** 2, axis=1)
                                    for center in centers]), axis=0)
        centers.append(signatures[int(rng.choice(len(signatures),
                                                  p=distance / distance.sum()))]
                       if distance.sum() > 0 else signatures[int(rng.integers(len(signatures)))])
    centers = np.stack(centers)
    for _ in range(8):
        assignments = np.argmin(np.sum((signatures[:, None] - centers[None]) ** 2, axis=2), axis=1)
        for code in range(n_groups):
            if np.any(assignments == code):
                centers[code] = signatures[assignments == code].mean(0)
    order = np.arange(len(windows))
    for _ in range(epochs):
        rng.shuffle(order)
        for group in np.array_split(order, max(1, int(np.ceil(len(order) / 8)))):
            terms = []
            for code in np.unique(assignments[group]):
                indices = np.concatenate([windows[int(i)] for i in group
                                          if assignments[int(i)] == code])
                prediction = model(x[indices], a[indices], int(code))
                terms.append((((prediction - y[indices]) / model.delta_scale) ** 2).sum()
                             / (len(indices) * model.state_dim))
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.eval()
    return model


def _predict_all(model: MechanismLoom, states: np.ndarray, actions: np.ndarray,
                 gains: np.ndarray | None = None) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(states, dtype=torch.float32)
        a = torch.as_tensor(actions, dtype=torch.float32)
        g = None if gains is None else torch.as_tensor(gains, dtype=torch.float32)
        return np.stack([model(x, a, code, g).numpy() for code in range(len(PROGRAMS))])


def _fit_dwell_weibull(completed: Sequence[int], censored: Sequence[int]) -> tuple[float, float]:
    """Discrete Weibull likelihood with right-censored terminal regimes."""
    best = (-np.inf, 16.0, 1.0)
    for scale in (8.0, 12.0, 18.0, 28.0, 42.0, 64.0, 96.0, 128.0):
        for shape in (0.75, 1.0, 1.5, 2.0, 3.0):
            likelihood = -sum(((duration - 1) / scale) ** shape
                              for duration in (*completed, *censored))
            likelihood += sum(np.log(max(hazard(duration - 1, scale, shape), 1e-12))
                              for duration in completed)
            if likelihood > best[0]:
                best = (likelihood, scale, shape)
    return best[1], best[2]


@dataclass(frozen=True)
class SwitchWeft:
    transition: np.ndarray
    dwell_scale: float
    dwell_shape: float
    drift_std: float = 0.012
    jump_rate: float = 0.025
    jump_std: float = 0.08

    @classmethod
    def fit(cls, model: MechanismLoom, sequences: Sequence[Batch]) -> "SwitchWeft":
        """Estimate a semi-Markov law from block-level public residuals.

        Decoding is deliberately coarse: a single noisy Hopper transition
        cannot establish a structural edit.  Terminal segments contribute
        survival likelihood rather than being silently discarded.
        """
        paths: list[np.ndarray] = []
        lengths: list[int] = []
        block = 8
        for states, actions, targets in sequences:
            predictions = _predict_all(model, states, actions)
            scale = model.delta_scale.numpy()
            emission = np.mean(((predictions - targets[None]) / scale) ** 2, axis=2)
            # Sum evidence across contiguous public transitions.  The last
            # block may be shorter; durations below use exact step counts.
            edges = np.r_[np.arange(0, len(states), block), len(states)]
            emission = np.stack([emission[:, left:right].sum(axis=1)
                                 for left, right in zip(edges[:-1], edges[1:])], axis=1)
            cost = np.full(len(PROGRAMS), 0.0)
            back = np.empty((emission.shape[1], len(PROGRAMS)), dtype=int)
            sorted_emission = np.sort(emission, axis=0)
            # A change must pay several blocks' typical code-separation cost.
            switch_cost = float(max(3 * np.median(sorted_emission[1]
                                                        - sorted_emission[0]), 1e-4))
            penalty = np.array([[0.0 if i == j else
                                 switch_cost * (1 + 0.25 * sum(a != b for a, b in
                                                               zip(PROGRAMS[i], PROGRAMS[j])))
                                 for j in range(len(PROGRAMS))] for i in range(len(PROGRAMS))])
            for t in range(emission.shape[1]):
                options = cost[:, None] + penalty
                back[t] = options.argmin(axis=0)
                cost = emission[:, t] + options.min(axis=0)
            path = np.empty(emission.shape[1], dtype=int)
            path[-1] = int(cost.argmin())
            for t in range(len(path) - 1, 0, -1):
                path[t - 1] = back[t, path[t]]
            paths.append(path)
            lengths.append(len(states))
        counts = np.ones((len(PROGRAMS), len(PROGRAMS))) * 0.2
        completed: list[int] = []
        censored: list[int] = []
        for path, length in zip(paths, lengths):
            boundaries = np.r_[0, block * (np.flatnonzero(np.diff(path)) + 1), length]
            completed.extend(np.diff(boundaries)[:-1].tolist())
            censored.append(int(np.diff(boundaries)[-1]))
            for left, right in zip(path[:-1], path[1:]):
                if left != right:
                    counts[left, right] += 1
        np.fill_diagonal(counts, 0.0)
        transition = counts / counts.sum(1, keepdims=True)
        # Survival through d-1 transitions is observed in both kinds of
        # segment; only completed segments add a switch hazard at d.
        dwell_scale, dwell_shape = _fit_dwell_weibull(completed, censored)
        return cls(transition, dwell_scale, dwell_shape)


@dataclass
class _Particle:
    code: int
    gains: np.ndarray
    age: int
    evolution_id: int
    weight: float


@dataclass(frozen=True)
class TideMemory:
    code: int
    gains: np.ndarray
    evolution_id: int
    source_end: int


class TheoryTide:
    """Approximate semi-Markov filter and benefit-gated memory adapter."""

    method_type = "in_house_theory_tide"
    capabilities = ("current_kernel", "future_law_samples", "adaptation", "memory_selection")

    def __init__(self, loom: MechanismLoom, switch: SwitchWeft | Sequence[SwitchWeft],
                 noise_std: float,
                 *, seed: int = 0, particles: int = 36,
                 probes: tuple[np.ndarray, np.ndarray] | None = None):
        if noise_std <= 0:
            raise ValueError("positive noise required")
        self.loom = loom
        self.switches = (switch,) if isinstance(switch, SwitchWeft) else tuple(switch)
        if not self.switches:
            raise ValueError("at least one learned evolution hypothesis required")
        if particles < len(PROGRAMS) * len(self.switches):
            raise ValueError("need at least one particle per program/evolution pair")
        self.noise_std = noise_std
        self.probes = probes
        self.rng = np.random.default_rng(seed)
        self.particles = [_Particle(i % len(PROGRAMS), np.ones(loom.action_dim), 0,
                                    (i // len(PROGRAMS)) % len(self.switches),
                                    1.0 / particles) for i in range(particles)]
        self._steps = 0

    def _predict(self, states: np.ndarray, actions: np.ndarray,
                 codes: np.ndarray, gains: np.ndarray) -> np.ndarray:
        result = np.empty_like(states, dtype=float)
        with torch.no_grad():
            for code in np.unique(codes):
                at = np.flatnonzero(codes == code)
                result[at] = self.loom(torch.as_tensor(states[at], dtype=torch.float32),
                                       torch.as_tensor(actions[at], dtype=torch.float32),
                                       int(code), torch.as_tensor(gains[at], dtype=torch.float32)).numpy()
        return result

    def _branch(self, particle: _Particle) -> _Particle:
        evolution = self.switches[particle.evolution_id]
        age = particle.age
        code = particle.code
        gains = particle.gains.copy()
        if self.rng.random() < hazard(age, evolution.dwell_scale, evolution.dwell_shape):
            code = int(self.rng.choice(len(PROGRAMS), p=evolution.transition[code]))
            age = 0
            gains += self.rng.normal(0, 0.03, len(gains))
        else:
            age += 1
            gains += self.rng.normal(0, evolution.drift_std, len(gains))
            n_jumps = int(self.rng.poisson(evolution.jump_rate))
            if n_jumps:
                gains += self.rng.normal(0, evolution.jump_std, (n_jumps, len(gains))).sum(0)
        return _Particle(code, np.clip(gains, 0.5, 1.5), age,
                         particle.evolution_id, particle.weight)

    def _resample(self) -> None:
        """BranchWitness: retain code groups and their probability mass."""
        groups: dict[tuple[int, int, int], list[_Particle]] = {}
        for particle in self.particles:
            # The same program at different ages can imply different future
            # switch probabilities despite the same current transition law.
            key = (particle.code, min(particle.age // 4, 4), particle.evolution_id)
            groups.setdefault(key, []).append(particle)
        masses = {key: sum(item.weight for item in members)
                  for key, members in groups.items()}
        allocation = {key: 1 for key in groups}
        for _ in range(len(self.particles) - len(groups)):
            key = max(groups, key=lambda k: masses[k] / (allocation[k] + 1))
            allocation[key] += 1
        result = []
        for key, members in groups.items():
            weights = np.array([member.weight for member in members])
            weights /= weights.sum()
            for index in self.rng.choice(len(members), size=allocation[key], p=weights):
                item = members[index]
                result.append(_Particle(item.code, item.gains.copy(), item.age,
                                        item.evolution_id, masses[key] / allocation[key]))
        self.particles = result

    def observe(self, fragment: Batch) -> None:
        states, actions, targets = (np.asarray(part, dtype=float) for part in fragment)
        if not (len(states) == len(actions) == len(targets)):
            raise ValueError("transition lengths differ")
        if states.shape[1:] != (self.loom.state_dim,) or actions.shape[1:] != (self.loom.action_dim,):
            raise ValueError("state/action dimensions differ from learned loom")
        for state, action, target in zip(states, actions, targets):
            if self._steps:
                self.particles = [self._branch(item) for item in self.particles]
            codes = np.array([p.code for p in self.particles])
            gains = np.stack([p.gains for p in self.particles])
            predicted = self._predict(np.repeat(state[None], len(codes), axis=0),
                                      np.repeat(action[None], len(codes), axis=0), codes, gains)
            scaled = (predicted - target) / np.maximum(self.loom.delta_scale.numpy(),
                                                       self.noise_std)
            likelihood = -0.5 * np.mean(scaled ** 2, axis=1) / 0.20
            log_weights = np.log(np.maximum([p.weight for p in self.particles], 1e-300)) + likelihood
            weights = np.exp(log_weights - log_weights.max())
            weights /= weights.sum()
            for particle, weight in zip(self.particles, weights):
                particle.weight = float(weight)
            self._resample()
            self._steps += 1

    def _draw(self) -> _Particle:
        weights = np.array([p.weight for p in self.particles])
        item = self.particles[int(self.rng.choice(len(self.particles), p=weights / weights.sum()))]
        return _Particle(item.code, item.gains.copy(), item.age, item.evolution_id, 1.0)

    def _embedding(self, probes: tuple[np.ndarray, np.ndarray],
                   codes: np.ndarray, gains: np.ndarray) -> np.ndarray:
        x, a = probes
        n = len(codes)
        states = np.tile(x, (n, 1))
        actions = np.tile(a, (n, 1))
        code_rows = np.repeat(codes, len(x))
        gain_rows = np.repeat(gains, len(x), axis=0)
        means = self._predict(states, actions, code_rows, gain_rows).reshape(n, -1)
        return np.concatenate((means, np.full_like(means, self.noise_std ** 2)), axis=1)

    def predict_kernel(self, probes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        codes = np.array([p.code for p in self.particles])
        gains = np.stack([p.gains for p in self.particles])
        weights = np.array([p.weight for p in self.particles])
        return weights @ self._embedding(probes, codes, gains)

    def sample_current_laws(self, probes: tuple[np.ndarray, np.ndarray],
                            n_samples: int) -> np.ndarray:
        """Draw the filtered current law without evolving it."""
        if n_samples < 2:
            raise ValueError("at least two samples required")
        particles = [self._draw() for _ in range(n_samples)]
        return self._embedding(probes, np.array([p.code for p in particles]),
                               np.stack([p.gains for p in particles]))

    def forecast_laws(self, horizons: tuple[int, ...], probes: tuple[np.ndarray, np.ndarray],
                      n_samples: int = 32) -> dict[int, np.ndarray]:
        return self.forecast_with_frozen(horizons, probes, n_samples)[1]

    def forecast_with_frozen(self, horizons: tuple[int, ...],
                             probes: tuple[np.ndarray, np.ndarray],
                             n_samples: int = 32) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Forecast and frozen-current reference from the same initial draws."""
        if not horizons or min(horizons) < 1 or n_samples < 2:
            raise ValueError("positive horizons and at least two samples required")
        sampled = [self._draw() for _ in range(n_samples)]
        frozen = self._embedding(probes, np.array([p.code for p in sampled]),
                                 np.stack([p.gains for p in sampled]))
        result: dict[int, np.ndarray] = {}
        for step in range(1, max(horizons) + 1):
            sampled = [self._branch(item) for item in sampled]
            if step in horizons:
                codes = np.array([p.code for p in sampled])
                gains = np.stack([p.gains for p in sampled])
                result[step] = self._embedding(probes, codes, gains)
        return frozen, result

    def snapshot(self, source_end: int) -> TideMemory:
        item = max(self.particles, key=lambda p: p.weight)
        return TideMemory(item.code, item.gains.copy(), item.evolution_id, source_end)

    def adapt(self, recent_data: Batch, memory: TideMemory | None = None,
              budget: int = 3) -> "TheoryTide":
        if budget < 1 or not len(recent_data[0]):
            raise ValueError("positive budget and nonempty recent data required")
        # All branches execute the same number of filter updates.  Only the
        # prior differs.  Calling code limits recent_data to the shared budget.
        updated = TheoryTide(self.loom, self.switches, self.noise_std,
                             seed=0, particles=len(self.particles), probes=self.probes)
        if memory is not None:
            # Contaminate only a quarter of the common prior.  A stale MAP
            # snapshot must not erase every alternative program and evolution
            # hypothesis before the equal-budget adaptation begins.
            for index in range(0, len(updated.particles), 4):
                updated.particles[index] = _Particle(memory.code, memory.gains.copy(), 0,
                                                     memory.evolution_id,
                                                     1.0 / len(updated.particles))
        updated.observe(tuple(part[:budget] for part in recent_data))
        return updated

    def select_memory(self, history: Batch, available_memories: list[TideMemory],
                      budget: int = 3) -> int | None:
        gains = self.estimate_memory_gains(history, available_memories, budget)
        if len(gains) == 1:
            return None
        winner = int(np.argmax(gains))
        return None if winner == 0 else winner - 1

    def estimate_memory_gains(self, history: Batch, available_memories: list[TideMemory],
                              budget: int = 3) -> np.ndarray:
        """Posterior risk plus a public prequential check, with null included."""
        if len(history[0]) < 1 or not available_memories:
            return np.zeros(len(available_memories) + 1)
        data = tuple(part[:budget] for part in history)
        probes = self.probes if self.probes is not None else (data[0], data[1])
        # Update a copy of the current belief with public data.  This is the
        # method's estimate of current-law uncertainty, not evaluator truth.
        reference = TheoryTide(self.loom, self.switches, self.noise_std,
                               seed=0, particles=len(self.particles), probes=probes)
        reference.particles = [_Particle(item.code, item.gains.copy(), item.age,
                                         item.evolution_id, item.weight)
                               for item in self.particles]
        reference._steps = self._steps
        reference.observe(data)
        n_means = len(probes[0]) * self.loom.state_dim
        reference_next = reference.forecast_laws((1,), probes, 64)[1].mean(axis=0)[:n_means]
        posterior_risks = []
        observed_risks = []
        cut = max(2, len(data[0]) // 2)
        train = tuple(part[:cut] for part in data)
        validation = tuple(part[cut:] for part in data)
        for memory in [None, *available_memories]:
            fitted = self.adapt(data, memory, budget=budget)
            predicted = fitted.forecast_laws((1,), probes, 64)[1].mean(axis=0)[:n_means]
            posterior_risks.append(float(np.mean((reference_next - predicted) ** 2)))
            if len(validation[0]):
                heldout = self.adapt(train, memory, budget=cut)
                step_risks = []
                for state, action, target in zip(*validation):
                    probe = (state[None], action[None])
                    rng_state = copy.deepcopy(heldout.rng.bit_generator.state)
                    forecast = heldout.forecast_laws((1,), probe, 32)[1].mean(axis=0)
                    heldout.rng.bit_generator.state = rng_state
                    step_risks.append(float(np.mean((forecast[:self.loom.state_dim]
                                                     - target) ** 2)))
                    heldout.observe((state[None], action[None], target[None]))
                observed_risks.append(np.asarray(step_risks))
        posterior_gain = posterior_risks[0] - np.asarray(posterior_risks)
        if observed_risks:
            improvements = observed_risks[0] - np.asarray(observed_risks)
            observed_gain = improvements.mean(axis=1)
            gains = 0.5 * posterior_gain + 0.5 * observed_gain
            # A one-sided 95% Student bound is deliberately more conservative
            # than the point estimate.  Consecutive transitions are dependent,
            # so this remains an abstention heuristic, not a coverage claim.
            count = len(validation[0])
            stderr = (improvements.std(axis=1, ddof=1) / np.sqrt(count)
                      if count > 2 else np.full(len(improvements), np.inf))
            margin = float(student_t.ppf(0.95, count - 1)) if count > 2 else np.inf
            gains[1:] = np.where((posterior_gain[1:] > 0)
                                 & (observed_gain[1:] - margin * stderr[1:] > 0),
                                 gains[1:], -np.inf)
            return gains
        return posterior_gain
