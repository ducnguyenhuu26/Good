"""Matched public-input predictive baselines; no oracle regime information."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PooledGaussianMLP(nn.Module):
    """Two-layer unconditional transition MLP on public observation and action."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256,
                 variance_floor: float = 1e-5, variance_ceiling: float = 1e2):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(nn.Linear(observation_dim + action_dim, hidden_dim, dtype=torch.float64),
                                 nn.SiLU(), nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
                                 nn.SiLU(), nn.Linear(hidden_dim, observation_dim, dtype=torch.float64))
        self.log_variance = nn.Parameter(torch.full((observation_dim,), -3., dtype=torch.float64))
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        for name, width in (("obs", observation_dim), ("action", action_dim), ("delta", observation_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(width, dtype=torch.float64))
            self.register_buffer(f"{name}_scale", torch.ones(width, dtype=torch.float64))

    @torch.no_grad()
    def fit_normalization(self, obs: Tensor, action: Tensor, next_obs: Tensor) -> None:
        if obs.ndim != 2 or obs.shape[1] != self.observation_dim or action.shape != (len(obs), self.action_dim) or next_obs.shape != obs.shape:
            raise ValueError("public reference batch mismatch")
        for name, value in (("obs", obs), ("action", action), ("delta", next_obs - obs)):
            getattr(self, f"{name}_mean").copy_(value.mean(0))
            getattr(self, f"{name}_scale").copy_(value.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        if obs.ndim != 2 or obs.shape[1] != self.observation_dim or action.shape != (len(obs), self.action_dim):
            raise ValueError("public transition batch mismatch")
        features = torch.cat(((obs - self.obs_mean) / self.obs_scale,
                              (action - self.action_mean) / self.action_scale), dim=1)
        mean = obs + self.net(features) * self.delta_scale + self.delta_mean
        variance = self.log_variance.exp().clamp(self.variance_floor, self.variance_ceiling).expand_as(mean)
        return mean, variance

    def log_prob(self, obs: Tensor, action: Tensor, next_obs: Tensor) -> Tensor:
        mean, variance = self(obs, action)
        if next_obs.shape != mean.shape:
            raise ValueError("next observation shape mismatch")
        return -.5 * ((next_obs - mean).square() / variance + variance.log() + math.log(2 * math.pi)).sum(1)
