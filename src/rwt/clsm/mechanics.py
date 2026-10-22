"""Mechanism-structured bounded physical forces for CLSM-R2."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .schemas import EnvSchema
from .support import DisjointPartition, bounded_physical_strength


def code_bits(code: Tensor) -> Tensor:
    if code.ndim != 1 or torch.any((code < 0) | (code > 7)):
        raise ValueError("code must be a batch of integers in 0..7")
    return ((code.long()[:, None] >> torch.arange(3, device=code.device)) & 1).to(torch.float64)


class MechanismForce(nn.Module):
    """Known physical pathways with a selected, disjoint global partition.

    The partition is a non-gradient buffer selected from training data by the
    finite support M-step. No evaluator group assignment enters this module.
    """

    def __init__(self, schema: EnvSchema, partition: Tensor | None = None):
        super().__init__()
        self.schema = schema
        if partition is None:
            partition = DisjointPartition.enumerate(schema.action_dim)[0]
        partition = DisjointPartition.validate(partition, schema.action_dim)
        self.register_buffer("group_mask", partition.to(torch.float64))
        self.register_buffer("B", torch.as_tensor(schema.actuator_to_dof.copy(), dtype=torch.float64))
        self.register_buffer("d_ref", torch.as_tensor(schema.nominal_damping.copy(), dtype=torch.float64))
        self.register_buffer("velocity_indices", torch.as_tensor(schema.joint_velocity_indices, dtype=torch.long))

    @torch.no_grad()
    def set_partition(self, partition: Tensor) -> None:
        canonical = DisjointPartition.validate(partition, self.schema.action_dim)
        self.group_mask.copy_(canonical.to(device=self.group_mask.device, dtype=self.group_mask.dtype))

    @property
    def W(self) -> Tensor:
        """Compatibility accessor for diagnostic code; columns are disjoint."""
        return self.group_mask

    def forward(self, obs: Tensor, action: Tensor, code: Tensor,
                xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return effective, actuator and damping effort, each [B,n_dof]."""
        n = obs.shape[0]
        if obs.ndim != 2 or obs.shape[1] != self.schema.observation_dim:
            raise ValueError("obs must be [B,observation_dim]")
        if action.shape != (n, self.schema.action_dim) or xi.shape != (n, 3) or code.shape != (n,):
            raise ValueError("batch dimensions do not match")
        bits = code_bits(code).to(device=obs.device, dtype=obs.dtype)
        motor_multiplier, damping_multiplier = bounded_physical_strength(
            xi, self.schema.motor_multiplier_bounds, self.schema.damping_multiplier_bounds)
        # Multipliers are exactly one when their mechanism bit is inactive.
        group_delta = (motor_multiplier - 1.0) * bits[:, :2]
        actuator_gain = 1.0 + group_delta @ self.group_mask.to(obs.dtype).T
        tau_a = (actuator_gain * action) @ self.B.to(obs.dtype).T
        damping_scale = 1.0 + bits[:, 2:3] * (damping_multiplier - 1.0)
        velocity = obs[:, self.velocity_indices]
        tau_d = -damping_scale * self.d_ref.to(obs.dtype) * velocity
        return tau_a + tau_d, tau_a, tau_d
