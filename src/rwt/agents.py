"""Stage 1.5 policies and online Bayesian filtering over oracle theories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rwt.dynamics import CandidateLibrary, DynamicsParameters
from rwt.environment import task_reward


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class BayesianTheoryFilter:
    """HMM-style filter with a small hazard prior for recurrence and changes.

    The uniform base distribution is over oracle candidate dynamics. Before
    each transition likelihood, the previous posterior is mixed with that
    base prior: p_t^-=(1-h)p_(t-1)+h p_0. This is an explicit change hazard,
    not access to the simulator's regime labels or change times.
    """

    candidates: tuple[DynamicsParameters, ...]
    noise_std: float
    hazard_rate: float = 0.05
    posterior: FloatArray = field(init=False, repr=False)
    base_prior: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("candidate library cannot be empty")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if not 0.0 <= self.hazard_rate <= 1.0:
            raise ValueError("hazard_rate must lie in [0, 1]")
        # A tiny likelihood floor makes the exact, zero-noise case well-defined.
        self.noise_std = max(float(self.noise_std), 1e-6)
        self.base_prior = np.full(len(self.candidates), 1.0 / len(self.candidates))
        self.reset()

    def reset(self) -> None:
        self.posterior = self.base_prior.copy()

    @property
    def transition_prior(self) -> FloatArray:
        """Predictive regime distribution before seeing the next transition."""

        return (1.0 - self.hazard_rate) * self.posterior + self.hazard_rate * self.base_prior

    @property
    def entropy(self) -> float:
        return _entropy(self.posterior)

    def _posterior_for_velocity(
        self,
        state: FloatArray,
        action: FloatArray,
        velocity: FloatArray,
        prior: FloatArray,
    ) -> FloatArray:
        means = _velocity_means(self.candidates, state, action)
        return _normalize_log_weights(
            np.log(np.maximum(prior, np.finfo(np.float64).tiny))
            - 0.5 * np.square((means - velocity[None, :]) / self.noise_std).sum(axis=1)
        )

    def update(self, state: FloatArray, action: FloatArray, next_state: FloatArray) -> FloatArray:
        """Apply transition prior, then likelihood of the observed velocity."""

        self.posterior = self._posterior_for_velocity(
            state, action, next_state[2:], self.transition_prior
        )
        return self.posterior.copy()

    def posterior_for_transition(
        self, state: FloatArray, action: FloatArray, velocity: FloatArray
    ) -> FloatArray:
        """Predict a posterior using the same transition prior as ``update``."""

        return self._posterior_for_velocity(state, action, velocity, self.transition_prior)

    def expected_parameters(self, prior: FloatArray | None = None) -> tuple[float, FloatArray, FloatArray]:
        probabilities = self.posterior if prior is None else prior
        friction = float(sum(p * candidate.friction for p, candidate in zip(probabilities, self.candidates)))
        matrix = sum(
            (p * candidate.action_matrix for p, candidate in zip(probabilities, self.candidates)),
            start=np.zeros((2, 2)),
        )
        force = sum(
            (p * np.asarray(candidate.external_force) for p, candidate in zip(probabilities, self.candidates)),
            start=np.zeros(2),
        )
        return friction, matrix, force

    def expected_information_gain(
        self,
        state: FloatArray,
        action: FloatArray,
        *,
        standard_noise: FloatArray,
    ) -> float:
        """Estimate I(theory; next velocity | state, action) by common-noise MC.

        ``standard_noise`` has shape ``(num_candidates, samples, 2)``. Reusing
        it across candidate actions reduces Monte Carlo noise in action ranking.
        """

        if standard_noise.ndim != 3 or standard_noise.shape[0] != len(self.candidates) or standard_noise.shape[2] != 2:
            raise ValueError("standard_noise must have shape (num_candidates, samples, 2)")
        return float(self.expected_information_gain_for_actions(
            state, np.asarray(action, dtype=np.float64)[None, :], standard_noise=standard_noise
        )[0])

    def expected_information_gain_for_actions(
        self,
        state: FloatArray,
        actions: FloatArray,
        *,
        standard_noise: FloatArray,
    ) -> FloatArray:
        """Vectorized common-noise EIG for a batch of action candidates."""

        count = len(self.candidates)
        if standard_noise.ndim != 3 or standard_noise.shape[0] != count or standard_noise.shape[2] != 2:
            raise ValueError("standard_noise must have shape (num_candidates, samples, 2)")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("actions must have shape (num_actions, 2)")
        prior = self.transition_prior
        means = np.stack([_velocity_means(self.candidates, state, action) for action in actions])
        # M actions x K generating theories x S samples x K likelihood theories.
        observations = means[:, :, None, :] + self.noise_std * standard_noise[None, :, :, :]
        residual = (
            observations[:, :, :, None, :] - means[:, None, None, :, :]
        ) / self.noise_std
        log_likelihood = -0.5 * np.square(residual).sum(axis=-1)
        log_weights = log_likelihood + np.log(np.maximum(prior, np.finfo(np.float64).tiny))[None, None, None, :]
        log_weights -= np.max(log_weights, axis=-1, keepdims=True)
        weights = np.exp(log_weights)
        posteriors = weights / weights.sum(axis=-1, keepdims=True)
        safe = np.maximum(posteriors, np.finfo(np.float64).tiny)
        posterior_entropy = -np.sum(np.where(posteriors > 0.0, posteriors * np.log(safe), 0.0), axis=-1)
        expected_entropy = np.sum(
            posterior_entropy * prior[None, :, None], axis=(1, 2)
        ) / standard_noise.shape[1]
        return np.maximum(0.0, _entropy(prior) - expected_entropy)


def _entropy(probabilities: FloatArray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _normalize_log_weights(log_weights: FloatArray) -> FloatArray:
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(log_weights), 1.0 / len(log_weights))
    return weights / total


def _velocity_means(
    candidates: tuple[DynamicsParameters, ...], state: FloatArray, action: FloatArray
) -> FloatArray:
    return np.stack([candidate.predict_velocity(state[2:], action) for candidate in candidates])


def constrain_action(action: FloatArray, max_norm: float) -> FloatArray:
    action = np.asarray(action, dtype=np.float64)
    norm = float(np.linalg.norm(action))
    if norm > max_norm:
        action = action * (max_norm / norm)
    return action


def probe_action_set(max_norm: float) -> FloatArray:
    """Shared finite probe set used by random, pure-IG and active policies."""

    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    angles = np.arange(8, dtype=np.float64) * (np.pi / 4.0)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    actions = [scale * max_norm * directions for scale in (1.0, 2.0 / 3.0, 1.0 / 3.0)]
    return np.concatenate(actions, axis=0)


def evenly_spaced_probe_steps(horizon: int, budget: int) -> frozenset[int]:
    """Use a fixed probe budget spread over public episode time, not change points."""

    if horizon < 1 or budget < 0:
        raise ValueError("horizon must be positive and budget non-negative")
    count = min(horizon, budget)
    if count == 0:
        return frozenset()
    return frozenset(np.linspace(0, horizon - 1, count, dtype=np.int64).tolist())


@dataclass
class PassiveBayesPolicy:
    """Bayesian transition inference with target feedback and no probe allocation."""

    candidates: tuple[DynamicsParameters, ...]
    noise_std: float
    seed: int = 0
    max_action_norm: float = 0.75
    horizon: int = 100
    probing_budget: int = 0
    hazard_rate: float = 0.05

    def __post_init__(self) -> None:
        self.filter = BayesianTheoryFilter(self.candidates, self.noise_std, self.hazard_rate)
        self.rng = np.random.default_rng(self.seed)
        self.probe_steps = evenly_spaced_probe_steps(self.horizon, self.probing_budget)
        self._probe_actions = probe_action_set(self.max_action_norm)

    def reset(self) -> None:
        self.filter.reset()
        self.rng = np.random.default_rng(self.seed)

    def observe(self, state: FloatArray, action: FloatArray, next_state: FloatArray, reward: float) -> None:
        del reward
        self.filter.update(state, action, next_state)

    def control_action(self, state: FloatArray, target: FloatArray) -> FloatArray:
        friction, matrix, force = self.filter.expected_parameters(self.filter.transition_prior)
        desired_velocity = state[2:] + 0.28 * (target - state[:2]) - 0.65 * state[2:]
        right_hand_side = desired_velocity - (1.0 - friction) * state[2:] - force
        return constrain_action(np.linalg.pinv(matrix) @ right_hand_side, self.max_action_norm)

    def feasible_actions(self, state: FloatArray, target: FloatArray) -> FloatArray:
        control = self.control_action(state, target)
        return np.concatenate((self._probe_actions, control[None, :]), axis=0)

    def act(self, state: FloatArray, target: FloatArray, step: int) -> FloatArray:
        del step
        return self.control_action(state, target)


@dataclass
class RandomProbePolicy(PassiveBayesPolicy):
    """Randomly selects from the shared feasible set on allocated probe steps."""

    def act(self, state: FloatArray, target: FloatArray, step: int) -> FloatArray:
        if step in self.probe_steps:
            feasible = self.feasible_actions(state, target)
            return feasible[int(self.rng.integers(len(feasible)))].copy()
        return self.control_action(state, target)


@dataclass
class PureInformationGainPolicy(PassiveBayesPolicy):
    """Maximizes expected theory information gain during its probe budget."""

    samples_per_candidate: int = 4

    def act(self, state: FloatArray, target: FloatArray, step: int) -> FloatArray:
        if step not in self.probe_steps:
            return self.control_action(state, target)
        feasible = self.feasible_actions(state, target)
        standard_noise = self.rng.standard_normal(
            (len(self.candidates), self.samples_per_candidate, 2)
        )
        gains = self.filter.expected_information_gain_for_actions(
            state, feasible, standard_noise=standard_noise
        )
        return feasible[int(np.argmax(gains))].copy()


@dataclass
class BeliefActivePolicy(PassiveBayesPolicy):
    """Maximizes expected task reward plus beta times expected information gain."""

    information_weight: float = 1.0
    samples_per_candidate: int = 4

    def act(self, state: FloatArray, target: FloatArray, step: int) -> FloatArray:
        if step not in self.probe_steps:
            return self.control_action(state, target)
        feasible = self.feasible_actions(state, target)
        standard_noise = self.rng.standard_normal(
            (len(self.candidates), self.samples_per_candidate, 2)
        )
        gains = self.filter.expected_information_gain_for_actions(
            state, feasible, standard_noise=standard_noise
        )
        scores = []
        for action, information_gain in zip(feasible, gains):
            means = _velocity_means(self.candidates, state, action)
            predictive_prior = self.filter.transition_prior
            candidate_rewards = [
                task_reward(
                    np.concatenate((state[:2] + velocity, velocity)), action, target
                )
                for velocity in means
            ]
            # The Gaussian noise contribution is constant across actions but
            # included so this is the actual one-step expected task reward.
            expected_reward = float(np.dot(predictive_prior, candidate_rewards))
            expected_reward -= 2.0 * self.filter.noise_std**2 * (1.0 + 0.08)
            scores.append(expected_reward + self.information_weight * information_gain)
        return feasible[int(np.argmax(scores))].copy()


def make_candidate_library() -> tuple[DynamicsParameters, ...]:
    return CandidateLibrary().parameters()
