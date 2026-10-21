"""Static, nominal-only MuJoCo topology and public observation maps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvSchema:
    env_id: str
    observation_dim: int
    action_dim: int
    joint_velocity_indices: tuple[int, ...]
    controlled_dof_indices: tuple[int, ...]
    actuator_to_dof: np.ndarray
    nominal_damping: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    dt_env: float
    public_reward_spec: str = "stationary_public_v1"

    def __post_init__(self) -> None:
        j = len(self.controlled_dof_indices)
        if j == 0 or len(self.joint_velocity_indices) != j:
            raise ValueError("each controlled DOF requires a public velocity")
        if self.actuator_to_dof.shape != (j, self.action_dim):
            raise ValueError("actuator topology shape mismatch")
        if self.nominal_damping.shape != (j,) or not np.any(self.nominal_damping > 0):
            raise ValueError("selected nominal damping must be positive on at least one DOF")
        if np.any(self.nominal_damping < 0):
            raise ValueError("negative reference damping")
        if max(self.joint_velocity_indices) >= self.observation_dim:
            raise ValueError("velocity mapping exceeds public observation")
        if self.action_low.shape != (self.action_dim,) or self.action_high.shape != (self.action_dim,):
            raise ValueError("action bounds mismatch")


def mujoco_nominal_schema(env_id: str) -> EnvSchema:
    """Inspect an untouched nominal model only. Call before intervention wrappers."""
    import gymnasium as gym
    import mujoco

    env = gym.make(env_id)
    try:
        obs, _ = env.reset(seed=0)
        model = env.unwrapped.model
        if not all(int(t) == int(mujoco.mjtTrn.mjTRN_JOINT) for t in model.actuator_trntype):
            raise ValueError(f"{env_id}: non-joint transmission needs a state-dependent adapter")
        if not all(int(t) == int(mujoco.mjtGain.mjGAIN_FIXED) for t in model.actuator_gaintype):
            raise ValueError(f"{env_id}: non-fixed gain is not a direct torque motor")
        if not all(int(t) == int(mujoco.mjtBias.mjBIAS_NONE) for t in model.actuator_biastype):
            raise ValueError(f"{env_id}: bias actuator requires a different adapter")
        dofs = []
        for joint_id in model.actuator_trnid[:, 0]:
            dofs.append(int(model.jnt_dofadr[int(joint_id)]))
        if len(set(dofs)) != len(dofs):
            raise ValueError("multiple actuator channels per DOF need explicit topology")
        # These offsets are checked against the installed environment's _get_obs,
        # not inferred by assuming obs[:nq] == qpos.
        if env_id in ("HalfCheetah-v5", "Walker2d-v5"):
            velocity_offset = model.nq - 1
        elif env_id == "Swimmer-v5":
            velocity_offset = model.nq - 2
        elif env_id == "Reacher-v5":
            velocity_offset = 6  # cos(2), sin(2), target qpos(2), then qvel[:2]
            if any(d >= 2 for d in dofs):
                raise ValueError("Reacher public observation omits a controlled velocity")
        else:
            raise ValueError(f"no verified public velocity map for {env_id}")
        velocity_indices = tuple(int(velocity_offset + d) for d in dofs)
        gear = np.asarray(model.actuator_gear[:, 0], dtype=np.float64)
        gain = np.asarray(model.actuator_gainprm[:, 0], dtype=np.float64)
        if np.any(gear == 0) or np.any(gain <= 0):
            raise ValueError("invalid nominal direct motor gain/gear")
        topology = np.diag(gear * gain)
        return EnvSchema(
            env_id, int(obs.size), int(model.nu), velocity_indices, tuple(dofs),
            topology, np.asarray(model.dof_damping[dofs], dtype=np.float64).copy(),
            np.asarray(env.action_space.low, dtype=np.float64).copy(),
            np.asarray(env.action_space.high, dtype=np.float64).copy(),
            float(env.unwrapped.dt),
        )
    finally:
        env.close()
