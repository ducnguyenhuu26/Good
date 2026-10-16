"""CPU-only diagnostic conditioning for HalfCheetah dynamics.

Oracle context is restricted to the evaluator. History context contains only
previously observed (action, state delta) pairs and an availability mask.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from rwt.stage3a import MiniDynamics


HISTORY_STEPS = 3
HISTORY_WIDTH = 6 + 17 + 1


def append_history(history: np.ndarray, state: np.ndarray, action: np.ndarray,
                   next_state: np.ndarray) -> np.ndarray:
    """Prepend one transition after it has been observed or predicted."""
    previous = np.asarray(history, dtype=np.float32)
    record = np.concatenate((np.asarray(action, np.float32),
                             np.asarray(next_state - state, np.float32),
                             np.ones(1, np.float32)))
    if previous.shape != (HISTORY_STEPS * HISTORY_WIDTH,) or record.shape != (HISTORY_WIDTH,):
        raise ValueError("history transition has wrong shape")
    return np.concatenate((record, previous[:-HISTORY_WIDTH]))


def observed_histories(data: dict[str, np.ndarray]) -> np.ndarray:
    """Context at row t uses rows strictly before t, resetting at episodes."""
    count = len(data["state"])
    result = np.zeros((count, HISTORY_STEPS * HISTORY_WIDTH), dtype=np.float32)
    current = np.zeros(HISTORY_STEPS * HISTORY_WIDTH, dtype=np.float32)
    for index in range(count):
        result[index] = current
        current = append_history(current, data["state"][index], data["action"][index],
                                 data["next_state"][index])
        if bool(data["terminated"][index] or data["truncated"][index]):
            current.fill(0)
    return result


class ConditionedMiniDynamics(MiniDynamics):
    """Same normalized residual/reward MLP with an explicit diagnostic context."""

    def __init__(self, context_dim: int, hidden_dim: int = 64):
        if context_dim < 1:
            raise ValueError("context_dim must be positive")
        super().__init__(17, 6, hidden_dim)
        self.context_dim = int(context_dim)
        self.net = nn.Sequential(nn.Linear(23 + context_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 18))
        self.register_buffer("context_mean", torch.zeros(context_dim))
        self.register_buffer("context_scale", torch.ones(context_dim))

    @torch.no_grad()
    def fit_context_normalization(self, context: np.ndarray) -> None:
        values = torch.as_tensor(context, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != self.context_dim:
            raise ValueError("context matrix has wrong shape")
        self.context_mean.copy_(values.mean(0))
        self.context_scale.copy_(values.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, state: Tensor, action: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        if (state.ndim != 2 or state.shape[1] != 17 or action.shape != (len(state), 6)
                or context.shape != (len(state), self.context_dim)):
            raise ValueError("conditioned dynamics input shapes do not align")
        x = torch.cat(((state - self.state_mean) / self.state_scale,
                       (action - self.action_mean) / self.action_scale,
                       (context - self.context_mean) / self.context_scale), dim=-1)
        output = self.net(x)
        delta = output[:, :17] * self.delta_scale + self.delta_mean
        reward = output[:, 17:] * self.reward_scale + self.reward_mean
        return state + delta, reward.squeeze(-1)

    def predict(self, state: Tensor, action: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(state, action, context)

    def loss(self, state: Tensor, action: Tensor, next_state: Tensor,
             reward: Tensor, context: Tensor) -> Tensor:
        predicted_next, predicted_reward = self(state, action, context)
        delta_loss = ((predicted_next - next_state) / self.delta_scale).square().mean()
        reward_loss = ((predicted_reward - reward) / self.reward_scale.squeeze()).square().mean()
        return delta_loss + reward_loss


class TrainSupport:
    """Nearest training input distance with a validation-only 99% reference."""

    def __init__(self, train_state: np.ndarray, train_action: np.ndarray,
                 val_state: np.ndarray, val_action: np.ndarray):
        state_mean = np.asarray(train_state, np.float32).mean(0)
        state_scale = np.asarray(train_state, np.float32).std(0).clip(1e-4)
        action_mean = np.asarray(train_action, np.float32).mean(0)
        action_scale = np.asarray(train_action, np.float32).std(0).clip(1e-4)
        self.state_mean, self.state_scale = state_mean, state_scale
        self.action_mean, self.action_scale = action_mean, action_scale
        self.train = self._features(train_state, train_action)
        self.train_norm2 = (self.train * self.train).sum(1)
        self.threshold = float(np.quantile(self.distance(val_state, val_action), 0.99))

    def _features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = np.atleast_2d(np.asarray(state, np.float32))
        action = np.atleast_2d(np.asarray(action, np.float32))
        if state.shape[1] != 17 or action.shape != (len(state), 6):
            raise ValueError("support input shapes do not align")
        return np.concatenate(((state - self.state_mean) / self.state_scale,
                               (action - self.action_mean) / self.action_scale), axis=1)

    def distance(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        query = self._features(state, action)
        distances = ((query * query).sum(1, keepdims=True) + self.train_norm2[None]
                     - 2 * query @ self.train.T)
        return np.sqrt(np.maximum(distances.min(1), 0))
