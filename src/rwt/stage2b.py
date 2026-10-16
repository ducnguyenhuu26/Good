"""Online structural disambiguation over seen dynamics and novel completions.

No agent-facing API in this module accepts regime names, change times, or
ground-truth parameters. Oracle models are isolated in an explicit diagnostic
factory; the principal learned factory consumes an unlabeled Stage 2A expert
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np
import torch

from rwt.dynamics import DynamicsParameters
from rwt.stage2a import IndependentExpertsDynamics, MonolithicDynamics
from rwt.agents import constrain_action, evenly_spaced_probe_steps, probe_action_set


@dataclass(frozen=True)
class PredictionSet:
    """Gaussian next-velocity predictions for a fixed set of hypotheses."""

    names: tuple[str, ...]
    means: np.ndarray  # [hypotheses, 2]
    scales: np.ndarray  # [hypotheses, 2]
    known_count: int
    completion_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        k = len(self.names)
        if self.means.shape != (k, 2) or self.scales.shape != (k, 2):
            raise ValueError("prediction means/scales must have shape (hypotheses, 2)")
        if np.any(self.scales <= 0) or not np.isfinite(self.means).all():
            raise ValueError("predictions must be finite with positive scales")
        if not 1 <= self.known_count <= k:
            raise ValueError("known_count must be in [1, hypothesis_count]")
        if any(i < self.known_count or i >= k for i in self.completion_indices):
            raise ValueError("completion indices must refer to novel hypotheses")


def enumerate_completions(
    seen_predictions: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    """Return ``G_i = F_j + F_k - F_i`` for every omitted anchor index.

    The first axis indexes seen theories; all remaining axes are preserved.
    This works for means over arbitrary shared state/action probe batches.
    """
    values = np.asarray(seen_predictions, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] != 3:
        raise ValueError("exactly three seen predictions are required")
    terms = []
    for omitted in range(3):
        j, k = (i for i in range(3) if i != omitted)
        terms.append(values[j] + values[k] - values[omitted])
    return np.stack(terms), ((1, 2, 0), (0, 2, 1), (0, 1, 2))


def _gaussian_log_likelihood(observation: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    z = (observation[None, :] - means) / scales
    return -0.5 * (z ** 2 + 2.0 * np.log(scales) + np.log(2.0 * np.pi)).sum(axis=-1)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def oracle_prediction_set(
    seen_models: tuple[DynamicsParameters, DynamicsParameters, DynamicsParameters],
    state: np.ndarray,
    action: np.ndarray,
    *,
    noise_std: float,
) -> PredictionSet:
    """Diagnostic-only exact A/B/D predictors plus all three completions."""
    if len(seen_models) != 3 or noise_std <= 0:
        raise ValueError("oracle diagnostic requires three models and positive noise")
    full = np.stack([model.predict_state(state, action) for model in seen_models])
    completions, _ = enumerate_completions(full)
    all_full = np.concatenate((full, completions), axis=0)
    scales = np.full((6, 2), noise_std, dtype=np.float64)
    return PredictionSet(
        names=("seen_0", "seen_1", "seen_2", "completion_0", "completion_1", "completion_2"),
        means=all_full[:, 2:].copy(), scales=scales, known_count=3,
        completion_indices=(3, 4, 5),
    )


def _model_predictions(model, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = torch.as_tensor(np.asarray(state, np.float32)[None, :])
    a = torch.as_tensor(np.asarray(action, np.float32)[None, :])
    with torch.no_grad():
        if isinstance(model, IndependentExpertsDynamics):
            means = model.predict_codes(s, a)[0].cpu().numpy().astype(np.float64)
            scales = model.log_scale.clamp(-6, 2).exp().cpu().numpy().astype(np.float64)[:, 2:]
            return means, scales
        if isinstance(model, MonolithicDynamics):
            mean, scale = model.predict(s, a)
            return mean.cpu().numpy().astype(np.float64), scale.cpu().numpy().astype(np.float64)[:, 2:]
    raise TypeError(f"unsupported learned dynamics model: {type(model).__name__}")


def learned_compositional_prediction_set(
    model: IndependentExpertsDynamics, state: np.ndarray, action: np.ndarray,
) -> PredictionSet:
    """Turn three unlabeled expert predictions into three novel completions."""
    seen, seen_scales = _model_predictions(model, state, action)
    completions, _ = enumerate_completions(seen)
    # Under the independent Gaussian approximation, variance of Fj + Fk - Fi
    # is the sum of the three expert variances. This is intentionally visible
    # as a calibration assumption in the Stage 2B report.
    comp_scales = np.stack([
        np.sqrt(sum(seen_scales[j] ** 2 for j in range(3))) for _ in range(3)
    ])
    return PredictionSet(
        names=("expert_0", "expert_1", "expert_2", "completion_0", "completion_1", "completion_2"),
        means=np.concatenate((seen[:, 2:], completions[:, 2:]), axis=0),
        scales=np.concatenate((seen_scales, comp_scales), axis=0),
        known_count=3, completion_indices=(3, 4, 5),
    )


def learned_expert_prediction_set(
    model: IndependentExpertsDynamics, state: np.ndarray, action: np.ndarray,
) -> PredictionSet:
    means, scales = _model_predictions(model, state, action)
    return PredictionSet(
        names=("expert_0", "expert_1", "expert_2"), means=means[:, 2:],
        scales=scales, known_count=3, completion_indices=(),
    )


def learned_monolithic_prediction_set(
    model: MonolithicDynamics, state: np.ndarray, action: np.ndarray,
) -> PredictionSet:
    means, scales = _model_predictions(model, state, action)
    return PredictionSet(
        names=("monolithic",), means=means[:, 2:], scales=scales,
        known_count=1, completion_indices=(),
    )


@dataclass
class OnlineTheoryFilter:
    """Causal HMM filter with a small recurrence memory over candidate theories."""

    names: tuple[str, ...]
    known_count: int
    completion_indices: tuple[int, ...]
    hazard: float = 0.05
    remember_threshold: float = 0.8
    use_memory: bool = True
    posterior: np.ndarray = field(init=False, repr=False)
    remembered: set[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.names or not 1 <= self.known_count <= len(self.names):
            raise ValueError("invalid hypothesis set")
        if not 0 <= self.hazard <= 1 or not 0 < self.remember_threshold <= 1:
            raise ValueError("invalid hazard or memory threshold")
        if any(i < self.known_count or i >= len(self.names) for i in self.completion_indices):
            raise ValueError("invalid completion indices")
        self.posterior = np.full(len(self.names), 1.0 / len(self.names))
        # Seen models are available from offline A/B/D training. A novel
        # completion joins memory only after online posterior confidence passes
        # the threshold.
        self.remembered = set(range(self.known_count))

    @property
    def entropy(self) -> float:
        positive = self.posterior[self.posterior > 0]
        return float(-np.sum(positive * np.log(positive)))

    def base_prior(self) -> np.ndarray:
        uniform = np.full(len(self.names), 1.0 / len(self.names))
        if not self.use_memory or not self.remembered:
            return uniform
        memory = np.zeros(len(self.names), dtype=np.float64)
        ids = sorted(self.remembered)
        memory[ids] = 1.0 / len(ids)
        return 0.7 * memory + 0.3 * uniform

    def predict_weights(self) -> np.ndarray:
        prior = (1.0 - self.hazard) * self.posterior + self.hazard * self.base_prior()
        return prior / prior.sum()

    def update(self, log_likelihood: np.ndarray) -> np.ndarray:
        """Consume one observed transition; no pre-transition mutation occurs."""
        likelihood = np.asarray(log_likelihood, dtype=np.float64)
        if likelihood.shape != self.posterior.shape or not np.isfinite(likelihood).all():
            raise ValueError("one finite log likelihood per hypothesis is required")
        log_posterior = np.log(self.predict_weights().clip(1e-300)) + likelihood
        log_posterior -= np.max(log_posterior)
        self.posterior = np.exp(log_posterior)
        self.posterior /= self.posterior.sum()
        best = int(np.argmax(self.posterior))
        if self.use_memory and self.posterior[best] >= self.remember_threshold:
            self.remembered.add(best)
        return self.posterior.copy()


def expected_information_gain(
    predictions: PredictionSet,
    prior: np.ndarray,
    standard_noise: np.ndarray,
) -> float:
    """Monte Carlo MI over candidate transition distributions.

    ``standard_noise`` is shared across actions by the caller. It has shape
    ``[K, samples, 2]`` and no ground-truth dynamics enter this calculation.
    """
    p = np.asarray(prior, dtype=np.float64)
    k = len(predictions.names)
    noise = np.asarray(standard_noise, dtype=np.float64)
    if p.shape != (k,) or noise.ndim != 3 or noise.shape[0] != k or noise.shape[2] != 2:
        raise ValueError("prior/noise shapes do not match prediction hypotheses")
    if np.any(p < 0) or p.sum() <= 0:
        raise ValueError("prior must be a nonnegative nonzero distribution")
    p = p / p.sum()
    samples = noise.shape[1]
    observations = predictions.means[:, None, :] + predictions.scales[:, None, :] * noise
    # [generating hypothesis, sample, evaluated hypothesis, velocity dim]
    z = (observations[:, :, None, :] - predictions.means[None, None, :, :]) / predictions.scales[None, None, :, :]
    log_likelihood = -0.5 * (z ** 2 + 2 * np.log(predictions.scales)[None, None, :, :] + np.log(2 * np.pi)).sum(-1)
    log_joint = log_likelihood + np.log(p.clip(1e-300))[None, None, :]
    log_norm = np.logaddexp.reduce(log_joint, axis=-1, keepdims=True)
    posterior = np.exp(log_joint - log_norm)
    entropy = -np.sum(posterior * np.log(posterior.clip(1e-300)), axis=-1)
    expected = float(np.sum(p[:, None] * entropy.mean(axis=1)))
    prior_entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return max(0.0, prior_entropy - expected)


def hypothesis_log_likelihoods(predictions: PredictionSet, next_velocity: np.ndarray) -> np.ndarray:
    observation = np.asarray(next_velocity, dtype=np.float64)
    if observation.shape != (2,):
        raise ValueError("next_velocity must have shape (2,)")
    return _gaussian_log_likelihood(observation, predictions.means, predictions.scales)


@dataclass
class OnlineObservation:
    predictions: PredictionSet
    prior: np.ndarray
    posterior: np.ndarray
    predicted_next_state: np.ndarray
    mse: float
    nll: float
    entropy: float
    surprise: float


class OnlineStructuralAgent:
    """Active/passive/random online inference without privileged schedule input."""

    def __init__(
        self,
        prediction_factory,
        *,
        names: tuple[str, ...],
        known_count: int,
        completion_indices: tuple[int, ...],
        policy: Literal["passive", "random", "active"],
        seed: int,
        horizon: int,
        probing_budget: int = 24,
        max_action_norm: float = 0.75,
        noise_samples: int = 8,
        hazard: float = 0.05,
        use_memory: bool = True,
    ) -> None:
        if policy not in {"passive", "random", "active"}:
            raise ValueError("policy must be passive, random, or active")
        if noise_samples < 1:
            raise ValueError("noise_samples must be positive")
        self.prediction_factory = prediction_factory
        self.policy = policy
        self.seed = int(seed)
        # Keep probe-action randomness paired across model classes even though
        # their Monte Carlo EIG banks contain different numbers of hypotheses.
        self.action_rng = np.random.default_rng(seed)
        self.noise_rng = np.random.default_rng(seed + 0x2B5EED)
        self.probe_steps = evenly_spaced_probe_steps(horizon, probing_budget) if policy != "passive" else frozenset()
        self.action_set = probe_action_set(max_action_norm)
        self.max_action_norm = float(max_action_norm)
        self.noise_samples = int(noise_samples)
        self.filter = OnlineTheoryFilter(
            names, known_count, completion_indices, hazard=hazard, use_memory=use_memory,
        )
        self.target = np.asarray((2.0, 2.0), dtype=np.float64)
        self.last_probe = False
        self.last_information_gain = 0.0
        self.last_probe_cost = 0.0
        self.last_prior = self.filter.predict_weights()

    def control_action(self, state: np.ndarray) -> np.ndarray:
        desired = 0.30 * (self.target - state[:2]) - 0.72 * state[2:]
        return constrain_action(desired, self.max_action_norm)

    def act(self, state: np.ndarray, step: int) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        control = self.control_action(state)
        self.last_probe = step in self.probe_steps
        self.last_information_gain = 0.0
        self.last_probe_cost = 0.0
        self.last_prior = self.filter.predict_weights().copy()
        if not self.last_probe:
            return control
        if self.policy == "random":
            feasible = np.concatenate((self.action_set, control[None, :]), axis=0)
            action = feasible[int(self.action_rng.integers(len(feasible)))].copy()
            self.last_probe_cost = float(np.linalg.norm(action - control))
            predictive = self.prediction_factory(state, action)
            common_noise = self.noise_rng.standard_normal((len(self.filter.names), self.noise_samples, 2))
            self.last_information_gain = expected_information_gain(predictive, self.last_prior, common_noise)
            return action
        # A common normal sample set is shared by candidate actions. The EIG
        # uses only learned/oracle *predictive distributions*, never the env.
        feasible = np.concatenate((self.action_set, control[None, :]), axis=0)
        k = len(self.filter.names)
        common_noise = self.noise_rng.standard_normal((k, self.noise_samples, 2))
        gains = []
        for action in feasible:
            predictions = self.prediction_factory(state, action)
            gains.append(expected_information_gain(predictions, self.last_prior, common_noise))
        best = int(np.argmax(gains))
        self.last_information_gain = float(gains[best])
        # A one-model monolithic baseline has no theory uncertainty, so it
        # does not spend a probe merely because all actions tie at zero gain.
        if self.last_information_gain <= 1e-12:
            return control
        action = feasible[best].copy()
        self.last_probe_cost = float(np.linalg.norm(action - control))
        return action

    def observe(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> OnlineObservation:
        """Score prior prediction, then apply likelihood update after observation."""
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)
        predictions = self.prediction_factory(state, action)
        prior = self.filter.predict_weights()
        likelihood = hypothesis_log_likelihoods(predictions, next_state[2:])
        mean_velocity = np.einsum("k,kd->d", prior, predictions.means)
        predicted_next = np.concatenate((state[:2] + mean_velocity, mean_velocity))
        nll = -_logsumexp(np.log(prior.clip(1e-300)) + likelihood)
        mse = float(np.mean((predicted_next - next_state) ** 2))
        surprise = float(nll)
        posterior = self.filter.update(likelihood)
        return OnlineObservation(
            predictions=predictions, prior=prior.copy(), posterior=posterior,
            predicted_next_state=predicted_next, mse=mse, nll=float(nll),
            entropy=self.filter.entropy, surprise=surprise,
        )


__all__ = [
    "OnlineTheoryFilter", "PredictionSet", "enumerate_completions",
    "expected_information_gain", "hypothesis_log_likelihoods",
    "learned_compositional_prediction_set", "learned_expert_prediction_set",
    "learned_monolithic_prediction_set", "oracle_prediction_set", "OnlineStructuralAgent",
]
