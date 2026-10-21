"""Evaluator-owned MuJoCo interventions; never import into the learner package."""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from rwt.clsm.schemas import EnvSchema, mujoco_nominal_schema


@dataclass(frozen=True)
class HiddenLaw:
    code: int
    motor_1: float = 1.0
    motor_2: float = 1.0
    damping: float = 1.0
    reset: bool = False

    def __post_init__(self):
        if not 0 <= self.code <= 7:
            raise ValueError("code outside three-mechanism masks")
        if not (0 < self.motor_1 <= 1 and 0 < self.motor_2 <= 1 and self.damping >= 1):
            raise ValueError("physical intervention sign/range invalid")


def stationary_public_reward(observation: np.ndarray, action: np.ndarray) -> float:
    """Declared reward r(o,a), independent of next state and hidden physics."""
    return -0.001 * float(np.square(observation).sum()) - 0.01 * float(np.square(action).sum())


def fixed_motor_groups(schema: EnvSchema) -> tuple[np.ndarray, np.ndarray]:
    n = schema.action_dim
    if n < 2:
        raise ValueError("at least two distinct motor groups required")
    return np.arange(0, n // 2), np.arange(n // 2, n)


class HiddenPhysicsWorld(gym.Wrapper):
    """Intervene on MuJoCo arrays before a step; advance hidden law afterward.

    `law_at` and `last_transition_law` are evaluator-only. They never enter
    observation, reward or returned info. The collector must discard native info.
    """

    def __init__(self, env_id: str, schedule: list[HiddenLaw], schema: EnvSchema | None = None):
        if not schedule:
            raise ValueError("nonempty schedule required")
        self.schema = mujoco_nominal_schema(env_id) if schema is None else schema
        if self.schema.env_id != env_id:
            raise ValueError("schema environment mismatch")
        super().__init__(gym.make(env_id))
        model = self.unwrapped.model
        if not all(int(t) == int(mujoco.mjtGain.mjGAIN_FIXED) for t in model.actuator_gaintype):
            raise ValueError("gain intervention requires fixed-gain motors")
        self._base_gain = model.actuator_gainprm.copy()
        self._base_damping = model.dof_damping.copy()
        self._groups = fixed_motor_groups(self.schema)
        self.schedule = tuple(schedule)
        self.position = 0
        self.last_transition_law: HiddenLaw | None = None
        self._restore()

    @property
    def law_at(self) -> HiddenLaw:
        return self.schedule[self.position]

    def _restore(self) -> None:
        model = self.unwrapped.model
        model.actuator_gainprm[:] = self._base_gain
        model.dof_damping[:] = self._base_damping
        mujoco.mj_forward(model, self.unwrapped.data)

    def _apply(self, law: HiddenLaw) -> None:
        self._restore()
        model = self.unwrapped.model
        if law.code & 1:
            model.actuator_gainprm[self._groups[0], 0] *= law.motor_1
        if law.code & 2:
            model.actuator_gainprm[self._groups[1], 0] *= law.motor_2
        if law.code & 4:
            model.dof_damping[list(self.schema.controlled_dof_indices)] *= law.damping
        mujoco.mj_forward(model, self.unwrapped.data)

    def reset(self, *, seed=None, options=None):
        self._restore()
        observation, _ = self.env.reset(seed=seed, options=options)
        self.position = 0
        self.last_transition_law = None
        self._apply(self.law_at)
        return observation, {}

    def step(self, action):
        if self.position >= len(self.schedule):
            raise RuntimeError("schedule exhausted; reset world")
        current = self.law_at
        previous_obs = self.unwrapped._get_obs().copy()
        next_obs, _, terminated, truncated, _ = self.env.step(action)
        reward = stationary_public_reward(previous_obs, np.asarray(action))
        self.last_transition_law = current
        self.position += 1
        if self.position < len(self.schedule):
            self._apply(self.law_at)
        else:
            truncated = True
        return next_obs, reward, terminated, truncated, {}

    def close(self):
        self._restore()
        super().close()
