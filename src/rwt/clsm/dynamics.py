"""Shared force-to-delta network and calibrated Gaussian law emission."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .mechanics import MechanismForce
from .schemas import EnvSchema


class ExecutableLaw(nn.Module):
    def __init__(self, schema: EnvSchema, hidden_dim: int = 256, variance_floor: float = 1e-5,
                 variance_ceiling: float = 1e2, delta_min: float = 0.12):
        super().__init__()
        self.schema = schema
        self.mechanics = MechanismForce(schema, delta_min=delta_min)
        dj = len(schema.controlled_dof_indices)
        self.net = nn.Sequential(
            nn.Linear(schema.observation_dim + dj, hidden_dim, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden_dim, schema.observation_dim, dtype=torch.float64),
        )
        self.log_variance = nn.Parameter(torch.full((schema.observation_dim,), -3.0, dtype=torch.float64))
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        if not 0 < self.variance_floor < self.variance_ceiling:
            raise ValueError("invalid emission variance bounds")
        for name, width in (("obs", schema.observation_dim), ("force", dj), ("delta", schema.observation_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(width, dtype=torch.float64))
            self.register_buffer(f"{name}_scale", torch.ones(width, dtype=torch.float64))

    @torch.no_grad()
    def fit_normalization(self, obs: Tensor, action: Tensor, next_obs: Tensor) -> None:
        """Call with training/reference data only; caller enforces the split."""
        n = len(obs)
        code = torch.zeros(n, dtype=torch.long, device=obs.device)
        theta = torch.zeros((n, 3), dtype=obs.dtype, device=obs.device)
        force = self.mechanics(obs, action, code, theta)[0]
        for name, values in (("obs", obs), ("force", force), ("delta", next_obs - obs)):
            getattr(self, f"{name}_mean").copy_(values.mean(0))
            getattr(self, f"{name}_scale").copy_(values.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, obs: Tensor, action: Tensor, code: Tensor, theta: Tensor) -> tuple[Tensor, Tensor]:
        force = self.mechanics(obs, action, code, theta)[0]
        features = torch.cat(((obs - self.obs_mean) / self.obs_scale,
                              (force - self.force_mean) / self.force_scale), dim=-1)
        delta = self.net(features) * self.delta_scale + self.delta_mean
        variance = self.log_variance.exp().clamp(self.variance_floor, self.variance_ceiling).expand_as(delta)
        return obs + delta, variance

    def log_prob(self, obs: Tensor, action: Tensor, next_obs: Tensor, code: Tensor, theta: Tensor) -> Tensor:
        mean, variance = self(obs, action, code, theta)
        if next_obs.shape != mean.shape:
            raise ValueError("next observation shape mismatch")
        return (-0.5 * ((next_obs - mean).square() / variance + variance.log() + math.log(2 * math.pi))).sum(-1)
