"""Learned compositional transition models for RWT Stage 2A.

The models operate only on continuous ``(state, action)`` inputs.  Regime
names and physics parameters are deliberately absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from models.program_execution import ProgramExecutor
from rwt.identifiability import hmm_log_evidence


# These are abstract bit-vectors, not labels for environment regimes. The last
# code is reserved for a composition omitted from the training likelihood.
TRAIN_THEORY_CODES = ((0, 0), (1, 0), (0, 1))
ALL_THEORY_CODES = (*TRAIN_THEORY_CODES, (1, 1))


class ContinuousExecutorAdapter(nn.Module):
    """Expose the upstream NEO residual executor as a continuous dynamics net."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        config = SimpleNamespace(
            num_state_tokens=1, state_dim=state_dim,
            num_action_tokens=1, action_dim=action_dim,
            transition_hidden_dim=hidden_dim, dropout=0.0,
        )
        self.executor = ProgramExecutor(config)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        # ProgramExecutor itself implements a FiLM-conditioned residual map.
        predicted_state = self.executor(state.unsqueeze(1), action.unsqueeze(1)).squeeze(1)
        return predicted_state - state


class CompositionalDynamics(nn.Module):
    """Gaussian transition model: base + selected learned residual primitives.

    The predicted target is ``delta_state = s_next - s``. The learned base and
    primitive networks have no environment parameters or ground-truth labels.
    Theory code (b1,b2) composes residuals additively.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 48):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.base = ContinuousExecutorAdapter(self.state_dim, self.action_dim, hidden_dim)
        self.primitives = nn.ModuleList(
            [ContinuousExecutorAdapter(self.state_dim, self.action_dim, hidden_dim) for _ in range(2)]
        )
        self.log_scale = nn.Parameter(torch.full((state_dim,), -2.5))

    def predict_codes(self, state: Tensor, action: Tensor) -> Tensor:
        """Return means for all four bit-coded theories, shape ``[B,4,D]``."""
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape (B,{self.state_dim})")
        if action.ndim != 2 or action.shape != (state.shape[0], self.action_dim):
            raise ValueError(f"action must have shape (B,{self.action_dim})")
        base = self.base(state, action)
        p1, p2 = (primitive(state, action) for primitive in self.primitives)
        return torch.stack((base, base + p1, base + p2, base + p1 + p2), dim=1)

    def predict(self, state: Tensor, action: Tensor, theory: tuple[int, int]) -> tuple[Tensor, Tensor]:
        try:
            index = ALL_THEORY_CODES.index(tuple(theory))
        except ValueError as error:
            raise ValueError(f"unknown theory code: {theory}") from error
        means = self.predict_codes(state, action)[:, index]
        scale = self.log_scale.clamp(-6.0, 2.0).exp().expand_as(means)
        return state + means, scale

    def log_prob_codes(self, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        """Diagonal-Gaussian log likelihood for all codes, shape ``[B,4]``."""
        means = self.predict_codes(state, action)
        target_delta = (next_state - state).unsqueeze(1)
        log_scale = self.log_scale.clamp(-6.0, 2.0).view(1, 1, -1)
        standardized = (target_delta - means) * torch.exp(-log_scale)
        return -0.5 * (standardized.square() + 2.0 * log_scale + np.log(2.0 * np.pi)).sum(-1)


class MonolithicDynamics(nn.Module):
    """Single learned Gaussian transition predictor baseline."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 48):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.network = ContinuousExecutorAdapter(state_dim, action_dim, hidden_dim)
        self.log_scale = nn.Parameter(torch.full((state_dim,), -2.5))

    def predict(self, state: Tensor, action: Tensor, theory: tuple[int, int] | None = None) -> tuple[Tensor, Tensor]:
        mean = state + self.network(state, action)
        return mean, self.log_scale.clamp(-6.0, 2.0).exp().expand_as(mean)


class IndependentExpertsDynamics(nn.Module):
    """Three latent, independent transition experts with equal parameter budget."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 28):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.experts = nn.ModuleList([ContinuousExecutorAdapter(state_dim, action_dim, hidden_dim) for _ in range(3)])
        self.log_scale = nn.Parameter(torch.full((3, state_dim), -2.5))

    def predict_codes(self, state: Tensor, action: Tensor) -> Tensor:
        deltas = torch.stack([expert(state, action) for expert in self.experts], dim=1)
        return state[:, None, :] + deltas

    def predict(self, state: Tensor, action: Tensor, theory: tuple[int, int] | None = None) -> tuple[Tensor, Tensor]:
        if theory is not None:
            index = TRAIN_THEORY_CODES.index(tuple(theory))
            mean = self.predict_codes(state, action)[:, index]
            scale = self.log_scale[index].clamp(-6.0, 2.0).exp().expand_as(mean)
            return mean, scale
        raise ValueError("independent experts require a learned expert posterior")

    def log_prob_codes(self, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        means = self.predict_codes(state, action)
        log_scale = self.log_scale.clamp(-6.0, 2.0).view(1, 3, -1)
        standardized = (next_state[:, None, :] - means) * torch.exp(-log_scale)
        return -0.5 * (standardized.square() + 2.0 * log_scale + np.log(2.0 * np.pi)).sum(-1)


class LinearStructuredDynamics(nn.Module):
    """Linear base plus typed state/velocity and action residual factors.

    The first residual can see velocity only and the second action only. This
    encodes a causal input-partition prior, not environment parameters or regime
    labels. The base predictor sees the full continuous state-action input.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        super().__init__()
        if state_dim != 4 or action_dim != 2:
            raise ValueError("the typed linear RWT baseline expects state_dim=4, action_dim=2")
        self.state_dim, self.action_dim = state_dim, action_dim
        self.base = nn.Linear(state_dim + action_dim, state_dim)
        self.velocity_primitive = nn.Linear(2, state_dim, bias=False)
        self.action_primitive = nn.Linear(action_dim, state_dim, bias=False)
        self.log_scale = nn.Parameter(torch.full((state_dim,), -2.5))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.ones(action_dim))

    @torch.no_grad()
    def fit_input_normalization(self, state: np.ndarray, action: np.ndarray) -> None:
        """Fit input scales on train transitions only; never reads labels or C."""
        state_t = torch.as_tensor(state, dtype=self.state_mean.dtype, device=self.state_mean.device)
        action_t = torch.as_tensor(action, dtype=self.action_mean.dtype, device=self.action_mean.device)
        self.state_mean.copy_(state_t.mean(dim=0))
        self.state_scale.copy_(state_t.std(dim=0, unbiased=False).clamp_min(1e-3))
        self.action_mean.copy_(action_t.mean(dim=0))
        self.action_scale.copy_(action_t.std(dim=0, unbiased=False).clamp_min(1e-3))

    def predict_codes(self, state: Tensor, action: Tensor) -> Tensor:
        normalized_state = (state - self.state_mean) / self.state_scale
        normalized_action = (action - self.action_mean) / self.action_scale
        base = self.base(torch.cat((normalized_state, normalized_action), dim=-1))
        velocity_effect = self.velocity_primitive(normalized_state[:, 2:])
        action_effect = self.action_primitive(normalized_action)
        return torch.stack((base, base + velocity_effect, base + action_effect,
                            base + velocity_effect + action_effect), dim=1)

    def log_prob_codes(self, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        means = state[:, None, :] + self.predict_codes(state, action)
        log_scale = self.log_scale.clamp(-6.0, 2.0).view(1, 1, -1)
        standardized = (next_state[:, None, :] - means) * torch.exp(-log_scale)
        return -0.5 * (standardized.square() + 2.0 * log_scale + np.log(2.0 * np.pi)).sum(-1)

    def predict(self, state: Tensor, action: Tensor, theory: tuple[int, int]) -> tuple[Tensor, Tensor]:
        index = ALL_THEORY_CODES.index(tuple(theory))
        mean = state + self.predict_codes(state, action)[:, index]
        scale = self.log_scale.clamp(-6.0, 2.0).exp().expand_as(mean)
        return mean, scale


@dataclass
class TransitionArrays:
    """Agent-facing training data; contains transitions only, no labels."""

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray

    def __post_init__(self) -> None:
        n = self.state.shape[0]
        if self.state.ndim != 2 or self.action.ndim != 2 or self.next_state.ndim != 2:
            raise ValueError("transition arrays must be rank 2")
        if self.action.shape[0] != n or self.next_state.shape[0] != n:
            raise ValueError("transition arrays must have equal lengths")


def gaussian_log_prob(next_state: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    z = (next_state - mean) / scale
    return -0.5 * (z.square() + 2.0 * scale.log() + np.log(2.0 * np.pi)).sum(-1)


def mixture_nll(log_probs: Tensor) -> Tensor:
    """NLL of an equal-prior latent theory mixture, averaged over samples."""
    if log_probs.ndim != 2 or log_probs.shape[1] < 1:
        raise ValueError("log_probs must have shape (batch, theories)")
    return -(torch.logsumexp(log_probs, dim=1) - np.log(log_probs.shape[1])).mean()


def train_latent_model(
    model: nn.Module,
    train: TransitionArrays,
    validation: TransitionArrays,
    *,
    model_kind: str,
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 2e-3,
    seed: int = 0,
    device: str = "cpu",
) -> list[dict[str, float]]:
    """Fit by marginal likelihood; only three training theories are available.

    The validation set is likewise restricted to A/B/D schedules. Regime names
    are not arguments and cannot enter the optimizer or model-selection code.
    """
    if model_kind not in {"compositional", "experts", "monolithic", "linear"}:
        raise ValueError("unknown model_kind")
    torch.manual_seed(seed)
    model.to(device)
    tensors = lambda data: tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in (data.state, data.action, data.next_state))
    train_t = tensors(train)
    val_t = tensors(validation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train.state), generator=generator)
        train_loss = 0.0
        for ids_cpu in order.split(batch_size):
            ids = ids_cpu.to(device)
            s, a, sn = (x[ids] for x in train_t)
            optimizer.zero_grad(set_to_none=True)
            if model_kind in {"compositional", "linear"}:
                logp = model.log_prob_codes(s, a, sn)[:, :3]
            elif model_kind == "experts":
                logp = model.log_prob_codes(s, a, sn)
            else:
                mu, scale = model.predict(s, a)
                logp = gaussian_log_prob(sn, mu, scale)[:, None]
            loss = mixture_nll(logp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(ids_cpu)
        model.eval()
        with torch.no_grad():
            s, a, sn = val_t
            if model_kind in {"compositional", "linear"}:
                val_logp = model.log_prob_codes(s, a, sn)[:, :3]
            elif model_kind == "experts":
                val_logp = model.log_prob_codes(s, a, sn)
            else:
                mu, scale = model.predict(s, a)
                val_logp = gaussian_log_prob(sn, mu, scale)[:, None]
            val_loss = float(mixture_nll(val_logp))
        history.append({"epoch": float(epoch + 1), "train_nll": train_loss / len(train.state), "validation_nll": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _sequence_log_probs(model: nn.Module, data: TransitionArrays, kind: str) -> Tensor:
    s, a, sn = (torch.as_tensor(x, dtype=torch.float32, device=next(model.parameters()).device)
                for x in (data.state, data.action, data.next_state))
    if kind in {"compositional", "linear"}:
        return model.log_prob_codes(s, a, sn)[:, :3]
    if kind == "experts":
        return model.log_prob_codes(s, a, sn)
    mu, scale = model.predict(s, a)
    return gaussian_log_prob(sn, mu, scale)[:, None]


def train_hmm_model(
    model: nn.Module,
    train_sequences: list[TransitionArrays],
    validation_sequences: list[TransitionArrays],
    *,
    model_kind: str,
    epochs: int = 40,
    learning_rate: float = 2e-3,
    hazard: float = 0.04,
    seed: int = 0,
    device: str = "cpu",
    calibration: TransitionArrays | None = None,
    calibration_weight: float = 1.0,
) -> list[dict[str, float]]:
    """Train sequence likelihood by exact HMM forward marginalization.

    The optional calibration is a separate weak-anchor experiment. It treats
    provided transitions as the known reference theory and clamps them to code
    00; no other regime labels are used in this objective.
    """
    if model_kind not in {"compositional", "experts", "monolithic", "linear"}:
        raise ValueError("unknown model_kind")
    if not train_sequences or not validation_sequences:
        raise ValueError("HMM training requires nonempty train and validation episodes")
    torch.manual_seed(seed); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    calibration_t = None
    if calibration is not None:
        calibration_t = tuple(torch.as_tensor(x, dtype=torch.float32, device=device)
                              for x in (calibration.state, calibration.action, calibration.next_state))
    history = []; best = float("inf"); best_state = None
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_steps = sum(len(seq.state) for seq in train_sequences)
        train_nll = -sum(hmm_log_evidence(_sequence_log_probs(model, seq, model_kind), hazard)
                         for seq in train_sequences) / total_steps
        loss = train_nll
        if calibration_t is not None:
            s, a, sn = calibration_t
            if model_kind in {"compositional", "linear", "experts"}:
                ll = model.log_prob_codes(s, a, sn)
            else:
                mu, scale = model.predict(s, a)
                ll = gaussian_log_prob(sn, mu, scale)[:, None]
            # The first theory is the calibrated reference code 00.
            loss = loss - calibration_weight * ll[:, 0].mean()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_steps = sum(len(seq.state) for seq in validation_sequences)
            val_nll = -sum(hmm_log_evidence(_sequence_log_probs(model, seq, model_kind), hazard)
                           for seq in validation_sequences) / val_steps
            val_value = float(val_nll)
            train_value = float(train_nll.detach())
        history.append({"epoch": float(epoch + 1), "train_nll": train_value, "validation_nll": val_value})
        if val_value < best:
            best = val_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_privileged_factor_diagnostic(
    model: CompositionalDynamics,
    train: TransitionArrays,
    train_regimes: np.ndarray,
    validation: TransitionArrays,
    validation_regimes: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 2e-3,
    seed: int = 0,
    device: str = "cpu",
) -> list[dict[str, float]]:
    """Label-supervised factorization upper bound, diagnostic only.

    It is intentionally separate from the proposed unlabeled method and must
    never be used to report its scores. Only A/B/D labels are accepted, while
    C remains held out for subsequent code-(1,1) diagnostics.
    """
    mapping = {"A": 0, "B": 1, "D": 2}
    if set(np.unique(train_regimes)) - mapping.keys() or set(np.unique(validation_regimes)) - mapping.keys():
        raise ValueError("privileged factor diagnostic accepts training labels A/B/D only")
    if len(train_regimes) != len(train.state) or len(validation_regimes) != len(validation.state):
        raise ValueError("diagnostic regime arrays must align with transitions")
    torch.manual_seed(seed)
    model.to(device)
    to_tensor = lambda data: tuple(torch.as_tensor(x, dtype=torch.float32, device=device) for x in (data.state, data.action, data.next_state))
    train_t, val_t = to_tensor(train), to_tensor(validation)
    train_codes = torch.as_tensor([mapping[str(x)] for x in train_regimes], dtype=torch.long, device=device)
    val_codes = torch.as_tensor([mapping[str(x)] for x in validation_regimes], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    best = float("inf"); best_state = None
    for epoch in range(epochs):
        model.train(); total = 0.0
        for ids_cpu in torch.randperm(len(train.state), generator=generator).split(batch_size):
            ids = ids_cpu.to(device)
            s, a, sn = (x[ids] for x in train_t)
            codes = train_codes[ids]
            means = model.predict_codes(s, a)[torch.arange(len(ids), device=device), codes]
            log_scale = model.log_scale.clamp(-6.0, 2.0)
            z = ((sn - s) - means) * torch.exp(-log_scale)
            loss = 0.5 * (z.square() + 2 * log_scale + np.log(2.0 * np.pi)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()) * len(ids_cpu)
        model.eval()
        with torch.no_grad():
            s, a, sn = val_t
            means = model.predict_codes(s, a)[torch.arange(len(s), device=device), val_codes]
            z = ((sn - s) - means) * torch.exp(-model.log_scale.clamp(-6.0, 2.0))
            val_nll = float(0.5 * (z.square() + 2 * model.log_scale.clamp(-6.0, 2.0) + np.log(2.0 * np.pi)).sum(-1).mean())
        history.append({"epoch": float(epoch + 1), "train_nll": total / len(train.state), "validation_nll": val_nll})
        if val_nll < best:
            best = val_nll
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


class TheoryBelief:
    """Online Bayesian filter over learned theory codes, updated post-transition."""

    def __init__(self, codes: Iterable[tuple[int, int]], hazard: float = 0.02):
        self.codes = tuple(codes)
        if not self.codes or not 0.0 <= hazard <= 1.0:
            raise ValueError("belief needs codes and a hazard in [0,1]")
        self.hazard = float(hazard)
        self.posterior = np.full(len(self.codes), 1.0 / len(self.codes))

    def predict_weights(self) -> np.ndarray:
        n = len(self.posterior)
        prior = (1.0 - self.hazard) * self.posterior + self.hazard / n
        return prior / prior.sum()

    def update(self, log_likelihood: np.ndarray) -> np.ndarray:
        values = np.asarray(log_likelihood, dtype=np.float64)
        if values.shape != self.posterior.shape:
            raise ValueError("likelihood shape does not match learned theories")
        logp = np.log(self.predict_weights().clip(1e-300)) + values
        logp -= np.max(logp)
        self.posterior = np.exp(logp)
        self.posterior /= self.posterior.sum()
        return self.posterior.copy()


def save_checkpoint(path: str | Path, model: nn.Module, *, model_kind: str, config: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_kind": model_kind, "config": config, "state_dict": model.state_dict()}, destination)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = dict(payload["config"])
    kind = payload["model_kind"]
    if kind == "compositional":
        model = CompositionalDynamics(**config)
    elif kind == "experts":
        model = IndependentExpertsDynamics(**config)
    elif kind == "monolithic":
        model = MonolithicDynamics(**config)
    elif kind == "linear":
        model = LinearStructuredDynamics(**config)
    else:
        raise ValueError(f"unsupported model kind: {kind}")
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    model.eval()
    return model, {"model_kind": kind, "config": config}


__all__ = [
    "ALL_THEORY_CODES", "TRAIN_THEORY_CODES", "CompositionalDynamics",
    "ContinuousExecutorAdapter", "IndependentExpertsDynamics", "MonolithicDynamics", "TheoryBelief",
    "LinearStructuredDynamics", "train_hmm_model",
    "TransitionArrays", "load_checkpoint", "mixture_nll", "save_checkpoint",
    "train_latent_model", "train_privileged_factor_diagnostic",
]
