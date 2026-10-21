"""Typed actuation and damping mechanisms before the shared dynamics network."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .schemas import EnvSchema


def code_bits(code: Tensor) -> Tensor:
    if code.ndim != 1 or torch.any((code < 0) | (code > 7)):
        raise ValueError("code must be a batch of integers in 0..7")
    return ((code.long()[:, None] >> torch.arange(3, device=code.device)) & 1).to(torch.float64)


class MechanismForce(nn.Module):
    def __init__(self, schema: EnvSchema, delta_min: float = 0.12):
        super().__init__()
        if delta_min <= 0:
            raise ValueError("positive structural separation required")
        self.schema = schema
        self.delta_min = float(delta_min)
        # Break the two otherwise identical unlabeled group initializations.
        # This local RNG does not consume the experiment's sampling stream.
        init_rng = torch.Generator().manual_seed(1729)
        initial_support = -1.0 + 0.01 * torch.randn(
            (schema.action_dim, 2), dtype=torch.float64, generator=init_rng)
        self.W_raw = nn.Parameter(initial_support)
        self.damping_shape = nn.Sequential(
            nn.Linear(schema.observation_dim, 32, dtype=torch.float64), nn.SiLU(),
            nn.Linear(32, len(schema.controlled_dof_indices), dtype=torch.float64),
        )
        self.register_buffer("B", torch.as_tensor(schema.actuator_to_dof.copy(), dtype=torch.float64))
        self.register_buffer("d_ref", torch.as_tensor(schema.nominal_damping.copy(), dtype=torch.float64))
        self.register_buffer("velocity_indices", torch.as_tensor(schema.joint_velocity_indices, dtype=torch.long))

    @property
    def W(self) -> Tensor:
        positive = F.softplus(self.W_raw)
        return positive / positive.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def regularizer(self, sparse_weight: float = 1e-3, overlap_weight: float = 1e-2) -> Tensor:
        w = self.W
        return sparse_weight * w.abs().sum() + overlap_weight * (w[:, 0] * w[:, 1]).abs().sum()

    def forward(self, obs: Tensor, action: Tensor, code: Tensor, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return effective, actuator and damping effort, each [B,n_joint]."""
        n = obs.shape[0]
        if obs.ndim != 2 or obs.shape[1] != self.schema.observation_dim:
            raise ValueError("obs must be [B,observation_dim]")
        if action.shape != (n, self.schema.action_dim) or theta.shape != (n, 3) or code.shape != (n,):
            raise ValueError("batch dimensions do not match")
        bits = code_bits(code).to(obs.dtype)
        # Mask before the nonlinear map so inactive theta has exactly zero gradient.
        active_theta = theta * bits
        motor_log = -bits[:, :2] * (self.delta_min + F.softplus(active_theta[:, :2]))
        motor_gain = torch.exp(motor_log @ self.W.to(obs.dtype).T)
        tau_a = (motor_gain * action) @ self.B.to(obs.dtype).T
        positive = F.softplus(self.damping_shape(obs))
        valid = (self.d_ref > 0).to(obs.dtype)
        shape = positive / ((positive * valid).sum(dim=1, keepdim=True) / valid.sum()).clamp_min(1e-8)
        damping_multiplier = torch.exp(bits[:, 2] * (self.delta_min + F.softplus(active_theta[:, 2])))
        tau_d = -damping_multiplier[:, None] * self.d_ref.to(obs.dtype) * shape * obs[:, self.velocity_indices]
        return tau_a + tau_d, tau_a, tau_d
