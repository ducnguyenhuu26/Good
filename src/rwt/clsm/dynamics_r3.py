"""CLSM-R3 intervention-resolved, nominal-anchored transition decoder.

The R2 `ExecutableLaw` is intentionally left unchanged. This version preserves
the nominal observation-delta decoder as a frozen anchor and feeds each
mechanism's intervention strength and force change to a small residual MLP.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .dynamics import ExecutableLaw
from .schemas import EnvSchema
from .support import DisjointPartition, bounded_physical_strength


class InterventionChannels(NamedTuple):
    nominal_force: Tensor
    eta: Tensor
    delta_tau: Tensor
    effective_multipliers: Tensor


def code_bits(code: Tensor) -> Tensor:
    if code.ndim != 1 or torch.any((code < 0) | (code > 7)):
        raise ValueError("code must be a batch of integers in 0..7")
    return ((code.long()[:, None] >> torch.arange(3, device=code.device)) & 1).to(torch.float64)


def intervention_channels(schema: EnvSchema, partition: Tensor, obs: Tensor,
                          action: Tensor, code: Tensor, xi: Tensor) -> InterventionChannels:
    """Compute the frozen R3 physical channels from public inputs and hypothesis.

    `effective_multipliers` is exactly one on each inactive mechanism. `xi`
    affects an output only through its active bit, giving exactly zero inactive
    coordinate gradients.
    """
    if obs.ndim != 2 or obs.shape[1] != schema.observation_dim:
        raise ValueError("obs must be [batch, observation_dim]")
    n = len(obs)
    if action.shape != (n, schema.action_dim) or code.shape != (n,) or xi.shape != (n, 3):
        raise ValueError("input batch shapes do not match the declared schema")
    G = DisjointPartition.validate(partition, schema.action_dim).to(device=obs.device, dtype=obs.dtype)
    B = torch.as_tensor(schema.actuator_to_dof.copy(), device=obs.device, dtype=obs.dtype)
    dref = torch.as_tensor(schema.nominal_damping.copy(), device=obs.device, dtype=obs.dtype)
    vel_idx = torch.as_tensor(schema.joint_velocity_indices, device=obs.device, dtype=torch.long)
    bits = code_bits(code).to(device=obs.device, dtype=obs.dtype)
    motor, damping = bounded_physical_strength(xi, schema.motor_multiplier_bounds,
                                                schema.damping_multiplier_bounds)
    physical = torch.cat((motor[:, :2], damping), dim=1)
    eta = bits * (physical - 1.0)
    effective = 1.0 + eta
    velocity = obs.index_select(1, vel_idx)
    nominal_force = action @ B.T - velocity * dref
    d1 = (action * G[:, 0]) @ B.T * eta[:, 0:1]
    d2 = (action * G[:, 1]) @ B.T * eta[:, 1:2]
    d3 = -velocity * dref * eta[:, 2:3]
    delta_tau = torch.cat((d1, d2, d3), dim=1)
    return InterventionChannels(nominal_force, eta, delta_tau, effective)


class InterventionResolvedLaw(nn.Module):
    """Version-3 Gaussian transition law with an exact nominal anchor.

    The trainable residual sees normalized `h=(obs, action, nominal_force)` and
    normalized mechanism-resolved `c=(eta, delta_tau)`. Its output is a
    standardized observation-delta correction. The frozen `f0` is an
    independently nominal-trained R2 `ExecutableLaw`.
    """

    architecture_revision = 3
    architecture_id = "clsm_r3_intervention_resolved_nominal_anchored"

    def __init__(self, schema: EnvSchema, f0: ExecutableLaw, partition: Tensor,
                 residual_width: int, delta_mean: Tensor, delta_scale: Tensor,
                 variance_floor: float = 1e-5, variance_ceiling: float = 1e2):
        super().__init__()
        if f0.schema.env_id != schema.env_id:
            raise ValueError("nominal anchor schema/environment mismatch")
        if residual_width < 1:
            raise ValueError("residual width must be positive")
        self.schema = schema
        self.f0 = f0
        for param in self.f0.parameters():
            param.requires_grad_(False)
        self.f0.eval()
        partition = DisjointPartition.validate(partition, schema.action_dim)
        self.register_buffer("partition", partition.to(torch.float64).clone())
        self.register_buffer("delta_mean", torch.as_tensor(delta_mean, dtype=torch.float64).clone())
        self.register_buffer("delta_scale", torch.as_tensor(delta_scale, dtype=torch.float64).clone())
        if self.delta_mean.shape != (schema.observation_dim,) or self.delta_scale.shape != (schema.observation_dim,):
            raise ValueError("delta normalization shape does not match observation")
        if torch.any(self.delta_scale <= 0):
            raise ValueError("delta scale must be positive")
        h_dim = schema.observation_dim + schema.action_dim + len(schema.controlled_dof_indices)
        c_dim = 3 + 3 * len(schema.controlled_dof_indices)
        self.register_buffer("h_mean", torch.zeros(h_dim, dtype=torch.float64))
        self.register_buffer("h_scale", torch.ones(h_dim, dtype=torch.float64))
        self.register_buffer("c_mean", torch.zeros(c_dim, dtype=torch.float64))
        self.register_buffer("c_scale", torch.ones(c_dim, dtype=torch.float64))
        self.residual = nn.Sequential(
            nn.Linear(h_dim + c_dim, residual_width, dtype=torch.float64), nn.SiLU(),
            nn.Linear(residual_width, residual_width, dtype=torch.float64), nn.SiLU(),
            nn.Linear(residual_width, schema.observation_dim, dtype=torch.float64),
        )
        self.log_variance = nn.Parameter(torch.zeros(schema.observation_dim, dtype=torch.float64))
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        if not 0 < self.variance_floor < self.variance_ceiling:
            raise ValueError("invalid standardized predictive variance limits")

    def train(self, mode: bool = True):
        super().train(mode)
        self.f0.eval()
        return self

    def _raw_inputs(self, obs: Tensor, action: Tensor, code: Tensor,
                    xi: Tensor) -> tuple[Tensor, Tensor, InterventionChannels]:
        channels = intervention_channels(self.schema, self.partition, obs, action, code, xi)
        h = torch.cat((obs, action, channels.nominal_force), dim=1)
        c = torch.cat((channels.eta, channels.delta_tau), dim=1)
        return h, c, channels

    @torch.no_grad()
    def fit_channel_normalization(self, obs: Tensor, action: Tensor, code: Tensor,
                                  xi: Tensor) -> None:
        """Fit channel moments on declared training examples only."""
        h, c, _ = self._raw_inputs(obs, action, code, xi)
        self.h_mean.copy_(h.mean(0))
        self.h_scale.copy_(h.std(0, unbiased=False).clamp_min(1e-5))
        self.c_mean.copy_(c.mean(0))
        self.c_scale.copy_(c.std(0, unbiased=False).clamp_min(1e-5))

    def normalized_inputs(self, obs: Tensor, action: Tensor, code: Tensor,
                          xi: Tensor) -> tuple[Tensor, Tensor]:
        h, c, _ = self._raw_inputs(obs, action, code, xi)
        return (h - self.h_mean) / self.h_scale, (c - self.c_mean) / self.c_scale

    def forward(self, obs: Tensor, action: Tensor, code: Tensor,
                xi: Tensor) -> tuple[Tensor, Tensor]:
        h_raw, c_raw, _ = self._raw_inputs(obs, action, code, xi)
        hn = (h_raw - self.h_mean) / self.h_scale
        cn = (c_raw - self.c_mean) / self.c_scale
        c0 = (-self.c_mean) / self.c_scale
        residual = self.residual(torch.cat((hn, cn), dim=1)) - self.residual(torch.cat((hn, c0.expand_as(cn)), dim=1))
        n = len(obs)
        zero_code = torch.zeros(n, dtype=torch.long, device=obs.device)
        zero_xi = torch.zeros((n, 3), dtype=obs.dtype, device=obs.device)
        with torch.no_grad():
            nominal_mean, _ = self.f0(obs, action, zero_code, zero_xi)
        base = (nominal_mean - obs - self.delta_mean) / self.delta_scale
        mean = base + residual
        variance = self.log_variance.exp().clamp(self.variance_floor, self.variance_ceiling).expand_as(mean)
        return mean, variance

    def log_prob(self, obs: Tensor, action: Tensor, next_obs: Tensor,
                 code: Tensor, xi: Tensor) -> Tensor:
        mean, variance = self(obs, action, code, xi)
        target = (next_obs - obs - self.delta_mean) / self.delta_scale
        return (-0.5 * ((target - mean).square() / variance + variance.log() +
                        math.log(2 * math.pi))).sum(-1)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def checkpoint_metadata(self) -> dict:
        return {"architecture_revision": 3, "architecture_id": self.architecture_id,
                "schema_env": self.schema.env_id, "role": "oracle_diagnostic",
                "nominal_anchor_frozen": True, "partition": self.partition.detach().cpu().tolist(),
                "trainable_parameter_count": self.trainable_parameter_count(),
                "f0_parameter_count_frozen": sum(p.numel() for p in self.f0.parameters()),
                "variance_limits_standardized": [self.variance_floor, self.variance_ceiling]}


def validate_r3_checkpoint(payload: dict) -> None:
    if payload.get("architecture_revision") != 3 or payload.get("architecture_id") != InterventionResolvedLaw.architecture_id:
        raise ValueError("R3 loader rejects R2, legacy, and incompatible architecture checkpoints")


def load_r3_state(model: InterventionResolvedLaw, payload: dict) -> None:
    validate_r3_checkpoint(payload)
    if payload.get("schema_env") != model.schema.env_id:
        raise ValueError("R3 checkpoint environment schema mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
