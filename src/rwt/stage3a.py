"""CPU-sized HalfCheetah world-model prototype for RWT Stage 3A.

The environment owns all regime metadata. Models, residual adapters and MPC
consume observations, actions, observed rewards and next observations only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PhysicsRegime:
    friction: float
    actuator_gain: float


DEFAULT_REGIMES = {
    "A": PhysicsRegime(1.0, 1.0),
    "B": PhysicsRegime(0.6, 1.0),
    "D": PhysicsRegime(1.0, 0.7),
    "C": PhysicsRegime(0.6, 0.7),
}
DEFAULT_ORDER = ("A", "B", "D", "C", "A", "C")


class RecurrentHalfCheetah(gym.Wrapper):
    """Switches two physical factors between transitions, without info leakage.

    ``current_regime`` and ``last_transition_regime`` are evaluator-only
    properties. They are never placed in observations or step ``info``.
    """

    def __init__(self, schedule: list[tuple[str, int]], regimes: dict[str, PhysicsRegime] | None = None):
        if not schedule or any(dwell <= 0 for _, dwell in schedule):
            raise ValueError("schedule must contain positive dwell times")
        self.regimes = dict(DEFAULT_REGIMES if regimes is None else regimes)
        if any(name not in self.regimes for name, _ in schedule):
            raise ValueError("unknown regime in schedule")
        super().__init__(gym.make("HalfCheetah-v5"))
        model = self.unwrapped.model
        self._base_friction = model.geom_friction.copy()
        self._base_gain = model.actuator_gainprm.copy()
        self.schedule = tuple((name, int(dwell)) for name, dwell in schedule)
        self.total_steps = sum(dwell for _, dwell in self.schedule)
        self._segment = 0
        self._within = 0
        self._steps = 0
        self.last_transition_regime: str | None = None
        self._restore_physics()

    @property
    def current_regime(self) -> str:
        return self.schedule[self._segment][0]

    def _restore_physics(self) -> None:
        model = self.unwrapped.model
        model.geom_friction[:] = self._base_friction
        model.actuator_gainprm[:] = self._base_gain

    def _apply_regime(self, name: str) -> None:
        regime = self.regimes[name]
        if regime.friction <= 0 or regime.actuator_gain <= 0:
            raise ValueError("physics multipliers must be positive")
        self._restore_physics()
        model = self.unwrapped.model
        model.geom_friction[:] = self._base_friction * regime.friction
        # HalfCheetah-v5 uses fixed-gain motor actuators (gainprm[:,0]=1).
        # This changes motor force without changing action-space bounds.
        model.actuator_gainprm[:, 0] = self._base_gain[:, 0] * regime.actuator_gain
        import mujoco
        mujoco.mj_forward(model, self.unwrapped.data)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._restore_physics()
        observation, info = self.env.reset(seed=seed, options=options)
        self._segment = self._within = self._steps = 0
        self.last_transition_regime = None
        self._apply_regime(self.current_regime)
        return observation, info

    def step(self, action):
        if self._steps >= self.total_steps:
            raise RuntimeError("reset before stepping beyond schedule")
        self.last_transition_regime = self.current_regime
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        self._within += 1
        if self._steps >= self.total_steps:
            truncated = True
        elif self._within == self.schedule[self._segment][1]:
            self._segment += 1
            self._within = 0
            self._apply_regime(self.current_regime)
        return observation, reward, terminated, truncated, info

    def close(self):
        self._restore_physics()
        super().close()


class MiniDynamics(nn.Module):
    """Shared 2x64 MLP predicting normalized state delta and observed reward."""

    def __init__(self, state_dim: int = 17, action_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.state_dim, self.action_dim, self.hidden_dim = int(state_dim), int(action_dim), int(hidden_dim)
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, state_dim + 1))
        for name, size in (("state", state_dim), ("action", action_dim),
                           ("delta", state_dim), ("reward", 1)):
            self.register_buffer(f"{name}_mean", torch.zeros(size))
            self.register_buffer(f"{name}_scale", torch.ones(size))

    @torch.no_grad()
    def fit_normalization(self, state: np.ndarray, action: np.ndarray,
                          next_state: np.ndarray, reward: np.ndarray):
        inputs = {"state": state, "action": action, "delta": next_state - state,
                  "reward": np.asarray(reward).reshape(-1, 1)}
        for name, value in inputs.items():
            value_t = torch.as_tensor(value, dtype=torch.float32)
            getattr(self, f"{name}_mean").copy_(value_t.mean(0))
            getattr(self, f"{name}_scale").copy_(value_t.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must be [batch,{self.state_dim}]")
        if action.shape != (len(state), self.action_dim):
            raise ValueError(f"action must be [batch,{self.action_dim}]")
        x = torch.cat(((state - self.state_mean) / self.state_scale,
                       (action - self.action_mean) / self.action_scale), dim=-1)
        output = self.net(x)
        delta = output[:, :self.state_dim] * self.delta_scale + self.delta_mean
        reward = output[:, self.state_dim:] * self.reward_scale + self.reward_mean
        return state + delta, reward.squeeze(-1)

    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Stage 2-style transition interface, with a learned reward head."""
        return self.forward(state, action)

    def loss(self, state: Tensor, action: Tensor, next_state: Tensor, reward: Tensor) -> Tensor:
        predicted_next, predicted_reward = self(state, action)
        delta_error = ((predicted_next - next_state) / self.delta_scale).square().mean()
        reward_error = ((predicted_reward - reward) / self.reward_scale.squeeze()).square().mean()
        return delta_error + reward_error


class LinearResidualAdapter:
    """Dimension-independent RLS correction to a frozen model's next state."""

    def __init__(self, state_dim: int, action_dim: int, *, ridge: float = 20.0,
                 state_mean=None, state_scale=None, action_mean=None, action_scale=None):
        if state_dim < 1 or action_dim < 1 or ridge <= 0:
            raise ValueError("invalid adapter dimensions or ridge")
        self.state_dim, self.action_dim = state_dim, action_dim
        self.feature_dim = 1 + state_dim + action_dim
        self.weights = np.zeros((self.feature_dim, state_dim), dtype=np.float64)
        self.covariance = np.eye(self.feature_dim, dtype=np.float64) / ridge
        self.state_mean = np.zeros(state_dim) if state_mean is None else np.asarray(state_mean, dtype=np.float64).copy()
        self.state_scale = np.ones(state_dim) if state_scale is None else np.asarray(state_scale, dtype=np.float64).copy()
        self.action_mean = np.zeros(action_dim) if action_mean is None else np.asarray(action_mean, dtype=np.float64).copy()
        self.action_scale = np.ones(action_dim) if action_scale is None else np.asarray(action_scale, dtype=np.float64).copy()
        self.updates = 0
        self._tensor_cache: dict[tuple[torch.device, torch.dtype], tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor]] = {}

    @classmethod
    def from_model(cls, model: MiniDynamics, *, ridge: float = 20.0):
        values = [getattr(model, name).detach().cpu().numpy() for name in
                  ("state_mean", "state_scale", "action_mean", "action_scale")]
        return cls(model.state_dim, model.action_dim, ridge=ridge,
                   state_mean=values[0], state_scale=values[1],
                   action_mean=values[2], action_scale=values[3])

    @property
    def parameter_count(self) -> int:
        return self.weights.size

    def features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        if state.shape != (self.state_dim,) or action.shape != (self.action_dim,):
            raise ValueError("adapter inputs have wrong shapes")
        return np.concatenate(([1.0], (state - self.state_mean) / self.state_scale,
                               (action - self.action_mean) / self.action_scale))

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.features(state, action) @ self.weights

    def predict_batch(self, state: Tensor, action: Tensor) -> Tensor:
        key = (state.device, state.dtype)
        cached = self._tensor_cache.get(key)
        if cached is None or cached[0] != self.updates:
            cached = (self.updates,
                      torch.as_tensor(self.state_mean, device=state.device, dtype=state.dtype),
                      torch.as_tensor(self.state_scale, device=state.device, dtype=state.dtype),
                      torch.as_tensor(self.action_mean, device=state.device, dtype=state.dtype),
                      torch.as_tensor(self.action_scale, device=state.device, dtype=state.dtype),
                      torch.as_tensor(self.weights.copy(), device=state.device, dtype=state.dtype))
            self._tensor_cache[key] = cached
        _, state_mean, state_scale, action_mean, action_scale, weights = cached
        feature = torch.cat((torch.ones((len(state), 1), device=state.device, dtype=state.dtype),
                             (state - state_mean) / state_scale,
                             (action - action_mean) / action_scale), dim=1)
        return feature @ weights

    def update(self, state: np.ndarray, action: np.ndarray, observed_residual: np.ndarray):
        feature = self.features(state, action)
        residual = np.asarray(observed_residual, dtype=np.float64)
        if residual.shape != (self.state_dim,) or not np.isfinite(residual).all():
            raise ValueError("observed residual has wrong shape or nonfinite values")
        projected = self.covariance @ feature
        denominator = 1.0 + feature @ projected
        if not np.isfinite(denominator) or denominator <= 1e-12:
            raise FloatingPointError("RLS covariance lost positive definiteness")
        gain = projected / denominator
        self.weights += np.outer(gain, residual - feature @ self.weights)
        self.covariance -= np.outer(gain, projected)
        self.covariance = (self.covariance + self.covariance.T) / 2
        if not np.isfinite(self.weights).all() or not np.isfinite(self.covariance).all():
            raise FloatingPointError("nonfinite RLS update")
        self.updates += 1


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 3
    candidates: int = 32
    iterations: int = 2
    elite_fraction: float = 0.2


class MiniCEM:
    """Action-constrained CEM using learned transition/reward heads only."""

    def __init__(self, model: MiniDynamics, action_low: np.ndarray, action_high: np.ndarray,
                 config: CEMConfig, *, seed: int = 0, adapter: LinearResidualAdapter | None = None,
                 diagnostics: bool = False):
        if (config.horizon < 1 or config.candidates < 2 or config.iterations < 1
                or not 0 < config.elite_fraction <= 1):
            raise ValueError("invalid CEM budget")
        self.model = model.eval()
        self.config = config
        self.adapter = adapter
        self.device = next(model.parameters()).device
        self.low = torch.as_tensor(action_low, device=self.device, dtype=torch.float32)
        self.high = torch.as_tensor(action_high, device=self.device, dtype=torch.float32)
        if (self.low.shape != (model.action_dim,) or self.high.shape != self.low.shape
                or not bool(torch.all(self.low <= self.high))):
            raise ValueError("action bounds do not match model")
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.last_runtime_seconds = 0.0
        self.diagnostics = diagnostics
        self.last_predicted_return = float("nan")
        self.last_plan_ood_fraction = float("nan")
        self.last_invalid_candidates = 0
        self.last_plan_actions: np.ndarray | None = None
        self.last_plan_states: np.ndarray | None = None
        self.last_plan_rewards: np.ndarray | None = None

    @torch.no_grad()
    def rollout(self, state: Tensor, sequences: Tensor) -> tuple[Tensor, Tensor]:
        if sequences.ndim != 3 or sequences.shape[1:] != (self.config.horizon, self.model.action_dim):
            raise ValueError("sequences must be [candidate,horizon,action_dim]")
        current = state.reshape(1, -1).expand(len(sequences), -1)
        states, rewards = [], []
        for t in range(self.config.horizon):
            action = sequences[:, t].clamp(self.low, self.high)
            next_state, reward = self.model(current, action)
            if self.adapter is not None:
                next_state = next_state + self.adapter.predict_batch(current, action)
            states.append(next_state)
            rewards.append(reward)
            current = next_state
        return torch.stack(states, dim=1), torch.stack(rewards, dim=1)

    @torch.no_grad()
    def _rollout_returns(self, state: Tensor, sequences: Tensor) -> Tensor:
        """Score all candidate sequences in one batch without storing states."""
        current = state.reshape(1, -1).expand(len(sequences), -1)
        total = torch.zeros(len(sequences), device=self.device, dtype=state.dtype)
        for t in range(self.config.horizon):
            action = sequences[:, t]
            next_state, reward = self.model(current, action)
            if self.adapter is not None:
                next_state = next_state + self.adapter.predict_batch(current, action)
            total = total + reward
            current = next_state
        return total

    @torch.no_grad()
    def act(self, state: np.ndarray) -> np.ndarray:
        start = perf_counter()
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        mean = torch.zeros((self.config.horizon, self.model.action_dim), device=self.device)
        std = torch.ones_like(mean)
        n_elite = max(2, int(self.config.candidates * self.config.elite_fraction))
        best_action = None
        best_plan = None
        invalid_total = torch.zeros((), device=self.device, dtype=torch.int64)
        for _ in range(self.config.iterations):
            sequences = (mean + std * torch.randn((self.config.candidates, *mean.shape),
                                                   generator=self.rng, device=self.device)).clamp(self.low, self.high)
            values = self._rollout_returns(state_t, sequences)
            invalid_total = invalid_total + (~torch.isfinite(values)).sum()
            values = torch.nan_to_num(values, nan=-1e12, posinf=-1e12, neginf=-1e12)
            elite = sequences[torch.topk(values, n_elite).indices]
            mean, std = elite.mean(0), elite.std(0, unbiased=False).clamp_min(0.05)
            best_plan = sequences[values.argmax()]
            best_action = best_plan[0]
        self.last_invalid_candidates = int(invalid_total.item())
        if self.diagnostics:
            states, rewards = self.rollout(state_t, best_plan[None])
            self.last_predicted_return = float(rewards.sum().item())
            normalized = (states - self.model.state_mean) / self.model.state_scale
            self.last_plan_ood_fraction = float((normalized.abs() > 6).any(dim=-1).float().mean().item())
            self.last_plan_actions = best_plan.cpu().numpy().copy()
            self.last_plan_states = states[0].cpu().numpy().copy()
            self.last_plan_rewards = rewards[0].cpu().numpy().copy()
        else:
            self.last_predicted_return = float("nan")
            self.last_plan_ood_fraction = float("nan")
            self.last_plan_actions = self.last_plan_states = self.last_plan_rewards = None
        action = best_action.cpu().numpy().astype(np.float64)
        if self.last_invalid_candidates >= self.config.candidates * self.config.iterations:
            action = np.zeros(self.model.action_dim, dtype=np.float64)
        # The host transfer synchronizes CUDA, so wall time includes completed
        # GPU kernels rather than just asynchronous launch overhead.
        self.last_runtime_seconds = perf_counter() - start
        return action


class BatchedMiniCEM:
    """Plan for independent evaluation seeds in one GPU model batch.

    Each seed owns its own NumPy random stream. Model rollouts combine the
    environment and candidate axes, then reshape back before elite selection.
    The environments themselves still step independently on CPU.
    """

    def __init__(self, model: MiniDynamics, action_low: np.ndarray, action_high: np.ndarray,
                 config: CEMConfig, seeds: list[int]):
        if (not seeds or config.horizon < 1 or config.candidates < 2
                or config.iterations < 1 or not 0 < config.elite_fraction <= 1):
            raise ValueError("invalid batched CEM configuration")
        self.model = model.eval()
        self.config = config
        self.device = next(model.parameters()).device
        self.low = torch.as_tensor(action_low, device=self.device, dtype=torch.float32)
        self.high = torch.as_tensor(action_high, device=self.device, dtype=torch.float32)
        if (self.low.shape != (model.action_dim,) or self.high.shape != self.low.shape
                or not bool(torch.all(self.low <= self.high))):
            raise ValueError("action bounds do not match model")
        self.rngs = [np.random.default_rng(int(seed)) for seed in seeds]
        self.last_runtime_seconds = 0.0
        self.last_invalid_candidates = np.zeros(len(seeds), dtype=np.int64)
        self.last_plan_ood_fraction = np.zeros(len(seeds), dtype=np.float64)
        self.last_predicted_return = np.zeros(len(seeds), dtype=np.float64)

    @torch.no_grad()
    def act(self, states: np.ndarray, adapters: list[LinearResidualAdapter],
            indices: list[int]) -> np.ndarray:
        count = len(indices)
        if count == 0 or states.shape != (count, self.model.state_dim) or len(adapters) != count:
            raise ValueError("batched states/adapters/indices do not align")
        start = perf_counter()
        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        # Transfer each seed's current adapter once per action search, not at
        # every candidate and horizon step.
        weights = torch.as_tensor(np.stack([adapter.weights for adapter in adapters]),
                                  device=self.device, dtype=torch.float32)
        state_mean = torch.as_tensor(np.stack([adapter.state_mean for adapter in adapters]),
                                     device=self.device, dtype=torch.float32)[:, None]
        state_scale = torch.as_tensor(np.stack([adapter.state_scale for adapter in adapters]),
                                      device=self.device, dtype=torch.float32)[:, None]
        action_mean = torch.as_tensor(np.stack([adapter.action_mean for adapter in adapters]),
                                      device=self.device, dtype=torch.float32)[:, None]
        action_scale = torch.as_tensor(np.stack([adapter.action_scale for adapter in adapters]),
                                       device=self.device, dtype=torch.float32)[:, None]
        horizon, candidates, action_dim = (self.config.horizon, self.config.candidates,
                                           self.model.action_dim)
        mean = torch.zeros((count, horizon, action_dim), device=self.device)
        std = torch.ones_like(mean)
        elite_count = max(2, int(candidates * self.config.elite_fraction))
        batch_indices = torch.arange(count, device=self.device)
        best = None
        best_ood = None
        best_return = None
        invalid_total = torch.zeros(count, device=self.device, dtype=torch.int64)
        for _ in range(self.config.iterations):
            noise = np.stack([self.rngs[i].standard_normal((candidates, horizon, action_dim),
                                                            dtype=np.float32) for i in indices])
            noise_t = torch.as_tensor(noise, device=self.device)
            sequences = (mean[:, None] + std[:, None] * noise_t).clamp(self.low, self.high)
            current = states_t[:, None].expand(count, candidates, self.model.state_dim)
            values = torch.zeros((count, candidates), device=self.device)
            ood_steps = torch.zeros((count, candidates), device=self.device)
            for t in range(horizon):
                action = sequences[:, :, t]
                next_state, reward = self.model(current.reshape(count * candidates, -1),
                                                action.reshape(count * candidates, -1))
                feature = torch.cat((torch.ones((count, candidates, 1), device=self.device),
                                     (current - state_mean) / state_scale,
                                     (action - action_mean) / action_scale), dim=-1)
                correction = torch.bmm(feature, weights)
                current = next_state.reshape(count, candidates, -1) + correction
                normalized = (current - self.model.state_mean) / self.model.state_scale
                ood_steps = ood_steps + (normalized.abs() > 6).any(dim=-1).float()
                values = values + reward.reshape(count, candidates)
            invalid_total = invalid_total + (~torch.isfinite(values)).sum(dim=1)
            values = torch.nan_to_num(values, nan=-1e12, posinf=-1e12, neginf=-1e12)
            elite_indices = torch.topk(values, elite_count, dim=1).indices
            elite = torch.gather(sequences, 1, elite_indices[:, :, None, None].expand(-1, -1, horizon, action_dim))
            mean = elite.mean(dim=1)
            std = elite.std(dim=1, unbiased=False).clamp_min(0.05)
            best_indices = values.argmax(dim=1)
            best = sequences[batch_indices, best_indices, 0]
            best_ood = ood_steps[batch_indices, best_indices] / horizon
            best_return = values[batch_indices, best_indices]
        actions = best.cpu().numpy().astype(np.float64)
        invalid_counts = invalid_total.cpu().numpy()
        ood_fractions = best_ood.cpu().numpy()
        predicted_returns = best_return.cpu().numpy()
        for row, index in enumerate(indices):
            self.last_invalid_candidates[index] = int(invalid_counts[row])
            self.last_plan_ood_fraction[index] = float(ood_fractions[row])
            self.last_predicted_return[index] = float(predicted_returns[row])
            if invalid_counts[row] >= candidates * self.config.iterations:
                actions[row] = 0.0
        self.last_runtime_seconds = perf_counter() - start
        return actions


def save_checkpoint(model: MiniDynamics, path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "state_dim": model.state_dim,
                "action_dim": model.action_dim, "hidden_dim": model.hidden_dim,
                "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[MiniDynamics, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = MiniDynamics(payload["state_dim"], payload["action_dim"], payload["hidden_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload["extra"]
