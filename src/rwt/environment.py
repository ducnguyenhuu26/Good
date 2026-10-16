"""Gymnasium-compatible continuous 2D system with hidden recurrent regimes."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rwt.dynamics import Regime, default_regimes
from rwt.schedule import RegimeSchedule


FloatArray = NDArray[np.float64]


def task_reward(state: FloatArray, action: FloatArray, target: FloatArray) -> float:
    """Shared reach-the-target reward used by simulation and active planning."""

    distance_sq = float(np.square(state[:2] - target).sum())
    speed_sq = float(np.square(state[2:]).sum())
    action_sq = float(np.square(action).sum())
    return -(distance_sq + 0.08 * speed_sq + 0.015 * action_sq)


class RecurrentDynamicsEnv(gym.Env[FloatArray, FloatArray]):
    """Reach a target while the physical transition law changes invisibly.

    Observations are ``[x, y, vx, vy]`` only. ``info`` is always empty: hidden
    regime names, transition parameters and change times are evaluator data.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        schedule: RegimeSchedule,
        *,
        regimes: dict[str, Regime] | None = None,
        noise_std: float = 0.025,
        horizon: int | None = None,
        target: tuple[float, float] = (2.0, 2.0),
        max_action_norm: float = 0.75,
        max_state_norm: float = 75.0,
    ) -> None:
        super().__init__()
        self._regimes = regimes or default_regimes()
        schedule.validate(self._regimes)
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if max_action_norm <= 0.0 or max_state_norm <= 0.0:
            raise ValueError("action and state norm limits must be positive")
        if horizon is not None and not 1 <= horizon <= schedule.horizon:
            raise ValueError("horizon must lie within the schedule")
        self._schedule = schedule
        self.noise_std = float(noise_std)
        self.max_action_norm = float(max_action_norm)
        self.max_state_norm = float(max_state_norm)
        self.horizon = schedule.horizon if horizon is None else int(horizon)
        self.target = np.asarray(target, dtype=np.float64)
        self.action_space = gym.spaces.Box(-self.max_action_norm, self.max_action_norm, shape=(2,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-1e6, 1e6, shape=(4,), dtype=np.float64)
        self._rng = np.random.default_rng()
        self._state = np.zeros(4, dtype=np.float64)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        if options and "initial_state" in options:
            state = np.asarray(options["initial_state"], dtype=np.float64)
            if state.shape != (4,):
                raise ValueError("initial_state must have shape (4,)")
            self._state = state.copy()
        else:
            self._state = np.asarray([-2.0, -1.5, 0.0, 0.0]) + self._rng.normal(0.0, 0.05, 4)
        self._step = 0
        return self._state.copy(), {}

    def step(self, action: FloatArray):
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action):
            raise ValueError("action must be a finite 2D vector within the action-space bounds")
        action_norm = float(np.linalg.norm(action))
        if action_norm > self.max_action_norm:
            action = action * (self.max_action_norm / action_norm)
        if self._step >= self.horizon:
            raise RuntimeError("step called after episode termination")
        regime_name = self._schedule.regime_name_at(self._step)
        parameters = self._regimes[regime_name].parameters
        next_velocity = parameters.predict_velocity(self._state[2:], action)
        next_velocity = next_velocity + self._rng.normal(0.0, self.noise_std, 2)
        self._state = np.concatenate((self._state[:2] + next_velocity, next_velocity))
        if not np.isfinite(self._state).all():
            raise FloatingPointError("environment state became non-finite")
        reward = task_reward(self._state, action, self.target)
        self._step += 1
        terminated = float(np.linalg.norm(self._state)) > self.max_state_norm
        truncated = self._step >= self.horizon and not terminated
        return self._state.copy(), float(reward), terminated, truncated, {}

    def close(self) -> None:
        return None
