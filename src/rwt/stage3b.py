"""Stage 3B pilot components: independent state dynamics and reward networks.

Oracle context is supplied only by diagnostic evaluators. The deployable
pooled/history models never accept regime IDs or physics parameters.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class StateDynamics(nn.Module):
    """Residual MLP trained solely on normalized state delta error."""

    def __init__(self, context_dim: int = 0, hidden_dim: int = 64):
        super().__init__()
        if context_dim < 0 or hidden_dim < 1:
            raise ValueError("invalid state model dimensions")
        self.state_dim, self.action_dim = 17, 6
        self.context_dim, self.hidden_dim = int(context_dim), int(hidden_dim)
        self.net = nn.Sequential(nn.Linear(23 + context_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 17))
        for name, width in (("state", 17), ("action", 6), ("delta", 17),
                            ("context", context_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(width))
            self.register_buffer(f"{name}_scale", torch.ones(width))

    @torch.no_grad()
    def fit_normalization(self, state: np.ndarray, action: np.ndarray,
                          next_state: np.ndarray, context: np.ndarray | None = None) -> None:
        if self.context_dim and context is None:
            raise ValueError("conditioned dynamics needs training context")
        values = {"state": state, "action": action, "delta": next_state - state}
        if self.context_dim:
            values["context"] = context
        for name, value in values.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim != 2 or tensor.shape[1] != getattr(self, f"{name}_mean").numel():
                raise ValueError(f"invalid {name} normalization array")
            getattr(self, f"{name}_mean").copy_(tensor.mean(0))
            getattr(self, f"{name}_scale").copy_(tensor.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, state: Tensor, action: Tensor,
                context: Tensor | None = None) -> Tensor:
        if state.ndim != 2 or state.shape[1] != 17 or action.shape != (len(state), 6):
            raise ValueError("state/action shapes do not align")
        features = [(state - self.state_mean) / self.state_scale,
                    (action - self.action_mean) / self.action_scale]
        if self.context_dim:
            if context is None or context.shape != (len(state), self.context_dim):
                raise ValueError("conditioned dynamics needs aligned context")
            features.append((context - self.context_mean) / self.context_scale)
        elif context is not None:
            raise ValueError("pooled dynamics must not receive context")
        delta = self.net(torch.cat(features, dim=-1)) * self.delta_scale + self.delta_mean
        return state + delta

    def loss(self, state: Tensor, action: Tensor, next_state: Tensor,
             context: Tensor | None = None) -> Tensor:
        return ((self(state, action, context) - next_state) / self.delta_scale).square().mean()


class RewardPredictor(nn.Module):
    """Independent reward predictor with its own trunk and train-only stats."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(23, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 1))
        for name, width in (("state", 17), ("action", 6), ("reward", 1)):
            self.register_buffer(f"{name}_mean", torch.zeros(width))
            self.register_buffer(f"{name}_scale", torch.ones(width))

    @torch.no_grad()
    def fit_normalization(self, state: np.ndarray, action: np.ndarray,
                          reward: np.ndarray) -> None:
        for name, value in (("state", state), ("action", action),
                            ("reward", np.asarray(reward).reshape(-1, 1))):
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim != 2 or tensor.shape[1] != getattr(self, f"{name}_mean").numel():
                raise ValueError(f"invalid {name} normalization array")
            getattr(self, f"{name}_mean").copy_(tensor.mean(0))
            getattr(self, f"{name}_scale").copy_(tensor.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != 17 or action.shape != (len(state), 6):
            raise ValueError("reward input shapes do not align")
        features = torch.cat(((state - self.state_mean) / self.state_scale,
                              (action - self.action_mean) / self.action_scale), dim=-1)
        return (self.net(features) * self.reward_scale + self.reward_mean).squeeze(-1)

    def loss(self, state: Tensor, action: Tensor, reward: Tensor) -> Tensor:
        return ((self(state, action) - reward) / self.reward_scale.squeeze()).square().mean()


class SeparatedWorldModel(nn.Module):
    """Adapter for the existing MPC interface; neither network shares weights."""

    def __init__(self, dynamics: StateDynamics, reward: RewardPredictor):
        super().__init__()
        if dynamics.context_dim != 0:
            raise ValueError("the existing MPC requires a pooled state model")
        self.dynamics, self.reward_model = dynamics, reward
        self.state_dim, self.action_dim = 17, 6

    @property
    def state_mean(self) -> Tensor:
        return self.dynamics.state_mean

    @property
    def state_scale(self) -> Tensor:
        return self.dynamics.state_scale

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.dynamics(state, action), self.reward_model(state, action)

    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(state, action)
