"""Public-history residual adaptation and predictive-risk memory for SCOPE-Law.

Memory selection is a chronological validation heuristic under dependent data.
It is not a confidence/coverage guarantee. Every candidate uses the same recent
fit transitions, ridge penalty, validation transitions, and one linear solve.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .scope_law import ScopeLaw, fit_effect_script
from .scope_policy import portfolio_features
from .scope_ensemble import ValidatedScriptEnsemble, calibration_weight


def fit_anchored_script(sequences, *, seed: int, epochs: int = 3, window: int = 8):
    """Fit on chronological prefixes; calibrate likelihood on unused suffixes."""
    fit, calibration = [], []
    for sequence in sequences:
        split = max(16, int(0.75 * len(sequence[0])))
        if split >= len(sequence[0]):
            raise ValueError("need at least 24 public transitions per sequence")
        fit.append(tuple(part[:split] for part in sequence))
        calibration.append(tuple(part[split:] for part in sequence))
    model, transition, prior = fit_effect_script(
        fit, seed=seed, epochs=epochs, window=window, anchored=True,
        warmup_epochs=20, m_step_repeats=4, validation_sequences=calibration)
    pooled, _, _ = fit_effect_script(
        fit, seed=seed+103, epochs=2, window=window, warmup_epochs=20,
        m_step_repeats=4, validation_sequences=calibration, validation_code=0)
    states, actions, targets = [np.concatenate([seq[i] for seq in calibration]) for i in range(3)]
    with torch.no_grad():
        anchored_prediction = torch.stack([model(torch.as_tensor(states), torch.as_tensor(actions), code)
                                  for code in range(9)]).mean(0).numpy()
        pooled_prediction = pooled(torch.as_tensor(states), torch.as_tensor(actions), 0).numpy()
        weight = calibration_weight(anchored_prediction, pooled_prediction, targets, model.delta_scale.numpy())
        ensemble = ValidatedScriptEnsemble(model, pooled, weight)
        prediction = weight*anchored_prediction + (1-weight)*pooled_prediction
        residual_scale = np.sqrt(np.mean((prediction - targets) ** 2, axis=0))
        ensemble.likelihood_scale.copy_(torch.as_tensor(np.maximum(
            residual_scale, 0.1 * model.delta_scale.numpy()), dtype=torch.float32))
    ensemble.fit_trace = {"anchored": model.fit_trace, "pooled": pooled.fit_trace,
                          "weight": weight}
    return ensemble, transition, prior


class ResidualPortfolioDecoder(torch.nn.Module):
    """Compositional decoder plus a causal, shared residual regression head."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.state_dim, self.action_dim = base.state_dim, base.action_dim
        self.rank = min(4, self.state_dim)
        projection = np.random.default_rng(771).normal(size=(self.state_dim, self.rank))
        projection, _ = np.linalg.qr(projection)
        self.register_buffer("projection", torch.as_tensor(projection, dtype=torch.float32))
        self.feature_dim = 1 + self.rank + self.action_dim * (1 + self.rank)
        self.register_buffer("coefficient", torch.zeros(self.feature_dim, self.state_dim))

    @property
    def delta_scale(self):
        return self.base.delta_scale

    @property
    def likelihood_scale(self):
        return self.base.likelihood_scale

    def features(self, states, actions):
        state = torch.clamp((states - self.base.state_mean) / self.base.state_scale, -5, 5)
        compressed = torch.tanh(state @ self.projection)
        cross = (compressed[:, :, None] * actions[:, None, :]).flatten(1)
        return torch.cat((torch.ones_like(actions[:, :1]), compressed, actions, cross), -1)

    def forward(self, states, actions, code, gains=None):
        correction = (self.features(states, actions) @ self.coefficient).clamp(-3, 3)
        return self.base(states, actions, code, gains) + self.delta_scale * correction


class CausalResidualMemory:
    """Online U(D,M) with explicit no-reuse and chronological risk selection."""

    def __init__(self, decoder: ResidualPortfolioDecoder, *, enabled: bool = True,
                 window: int = 48, interval: int = 16, ridge: float = 0.1):
        self.decoder = decoder
        self.enabled, self.window, self.interval, self.ridge = enabled, window, interval, ridge
        self.buffer = deque(maxlen=window)
        self.memories: list[tuple[int, np.ndarray]] = []
        self.time = 0
        self.last_error = np.zeros(decoder.state_dim)
        self.last_gain = 0.0
        self.last_reuse = False
        self.trace = []
        self.no_reuse_coefficient = np.zeros_like(decoder.coefficient.numpy())

    @property
    def context_dim(self):
        return self.decoder.rank * (self.decoder.action_dim + 2) + 3

    def context(self, state):
        d = self.decoder
        actions = np.vstack((np.zeros((1, d.action_dim)), np.eye(d.action_dim)))
        states = np.repeat(np.asarray(state, np.float32)[None], len(actions), 0)
        with torch.no_grad():
            response = (d.features(torch.as_tensor(states), torch.as_tensor(actions, dtype=torch.float32))
                        @ d.coefficient @ d.projection).numpy()
        responses = np.concatenate((response[0], (response[1:] - response[0]).ravel()))
        last = np.clip(self.last_error, -5, 5) @ d.projection.numpy()
        return np.concatenate((np.clip(responses, -5, 5), last,
                               [np.clip(self.last_gain, -2, 2), float(self.last_reuse),
                                min(len(self.buffer) / self.window, 1.0)])).astype(np.float32)

    def _adapt(self, x, y, prior):
        # Dual solve: computational adaptation budget is identical for memories
        # and no-reuse even when feature dimension exceeds recent sample size.
        residual = y - x @ prior
        return prior + x.T @ np.linalg.solve(
            x @ x.T + self.ridge * len(x) * np.eye(len(x)), residual)

    def observe(self, state, action, target, base_prediction):
        d = self.decoder
        with torch.no_grad():
            features = d.features(torch.as_tensor(state[None], dtype=torch.float32),
                                  torch.as_tensor(action[None], dtype=torch.float32)).numpy()[0]
        residual = (target - base_prediction) / d.delta_scale.numpy()
        self.last_error = residual.copy()
        self.buffer.append((features, residual))
        self.time += 1
        if len(self.buffer) == self.window and self.time % self.interval == 0:
            x, y = (np.stack([item[i] for item in self.buffer]) for i in range(2))
            cut = len(x) // 2
            zero = np.zeros_like(d.coefficient.numpy())
            scratch = self._adapt(x[:cut], y[:cut], zero)
            loss = lambda b: np.mean((y[cut:] - x[cut:] @ b) ** 2, axis=1)
            scratch_loss, base_loss = loss(scratch), loss(zero)
            scratch_gain = base_loss-scratch_loss
            scratch_margin = 2*scratch_gain.std(ddof=1)/np.sqrt(len(scratch_gain))
            best = scratch if scratch_gain.mean()>scratch_margin else zero
            self.no_reuse_coefficient = best.copy()
            no_reuse_loss = loss(best)
            best_loss = float(no_reuse_loss.mean())
            self.last_gain, self.last_reuse = 0.0, False
            eligible = [(end, b) for end, b in self.memories
                        if self.enabled and end < self.time - self.window]
            for end, prior in eligible:
                candidate = self._adapt(x[:cut], y[:cut], prior)
                candidate_loss = loss(candidate)
                gain = no_reuse_loss - candidate_loss
                heuristic_margin = 2 * gain.std(ddof=1) / np.sqrt(len(gain))
                if gain.mean() > heuristic_margin and candidate_loss.mean() < best_loss:
                    best, best_loss = candidate, candidate_loss.mean()
                    self.last_gain, self.last_reuse = float(gain.mean()), True
            d.coefficient.copy_(torch.as_tensor(best, dtype=torch.float32))
            self.trace.append({"time": self.time, "memory_candidates": len(eligible),
                               "selected_memory": self.last_reuse,
                               "estimated_memory_gain": self.last_gain,
                               "scratch_validation_risk": float(scratch_loss.mean()),
                               "no_reuse_validation_risk": float(no_reuse_loss.mean()),
                               "selected_validation_risk": float(best_loss)})
        if self.time % (2 * self.window) == 0:
            self.memories.append((self.time, d.coefficient.numpy().copy()))
            self.memories = self.memories[-4:]


class AdaptiveLawContext:
    """One causal clock for posterior filtering, residual memory, and policy."""

    def __init__(self, model, transition, prior, *, seed, particles=96, memory=True,
                 noise_std=0.02):
        self.decoder = ResidualPortfolioDecoder(model)
        self.memory = CausalResidualMemory(self.decoder, enabled=memory)
        self.filter = ScopeLaw(self.decoder, transition, noise_std, seed=seed,
                               particles=particles, evolution_prior=prior, kernel_noise_std=0.0)

    def context(self, state):
        belief = portfolio_features(self.filter, predict_next=self.filter._latest_time > 0)
        return np.concatenate((belief, self.memory.context(state)))

    def observe(self, state, action, target):
        # A prior predictive mean is computed before using target. The residual
        # head learns only the public next state, never evaluator latent labels.
        belief = portfolio_features(self.filter, predict_next=self.filter._latest_time > 0)
        with torch.no_grad():
            x, a = torch.as_tensor(state[None]), torch.as_tensor(action[None])
            weights = np.asarray([p.weight for p in self.filter.particles])
            gain = weights @ np.stack([p.gains for p in self.filter.particles]) / weights.sum()
            g = torch.as_tensor(gain[None], dtype=torch.float32)
            base = sum(float(belief[k]) * self.decoder.base(x, a, k, g).numpy()[0]
                       for k in range(9))
            correction = (self.decoder.features(x, a) @ self.decoder.coefficient).clamp(-3, 3).numpy()[0]
        nmse = float(np.mean(((target - base) / self.decoder.delta_scale.numpy() - correction) ** 2))
        self.filter.observe((state[None], action[None], target[None]))
        self.memory.observe(state, action, target, base)
        return nmse
