"""Belief-conditioned Soft Actor-Critic for SCOPE-Law control.

The actor and critics learn from public transitions and actual rewards. The
world-model portfolio supplies only a causal belief feature at action time;
the evaluator's latent variables are never policy inputs or reward targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .scope_law import SCOPE_PROGRAMS, ScopeLaw
from .scope_support import Batch, hazard
from .scope_conditioned import ZeroStartContextNetwork


BELIEF_DIM = len(SCOPE_PROGRAMS) + 10
ACTION_PORTFOLIO_DIM = 14


def belief_dimension(action_dim: int) -> int:
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    return len(SCOPE_PROGRAMS) + 2 * action_dim + 4


def action_portfolio_dimension(action_dim: int) -> int:
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    return 2 * (1 + 2 * action_dim)


def portfolio_action_features(filter_model: ScopeLaw, state: np.ndarray, *,
                              action_cost: float,
                              reward_mode: str,
                              target_velocity: float = 1.0,
                              action_limit: float = 0.7) -> np.ndarray:
    """Posterior mean and spread of rewards for a small public action portfolio.

    The seven candidates are zero plus positive/negative single-actuator
    commands. Scores use the fitted transition decoder and the current particle
    weights; the policy still learns from observed rewards through SAC.
    """
    if action_cost < 0 or not np.isfinite(action_cost):
        raise ValueError("action cost must be finite and nonnegative")
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (filter_model.model.state_dim,):
        raise ValueError("unexpected public state dimension")
    candidates = np.zeros((1 + 2 * filter_model.model.action_dim,
                           filter_model.model.action_dim), dtype=np.float32)
    for dimension in range(filter_model.model.action_dim):
        candidates[1 + 2 * dimension, dimension] = action_limit
        candidates[2 + 2 * dimension, dimension] = -action_limit
    particles = filter_model.particles
    weights = np.asarray([particle.weight for particle in particles], dtype=float)
    weights /= weights.sum()
    particle_scores = np.empty((len(particles), len(candidates)), dtype=float)
    for code in range(len(SCOPE_PROGRAMS)):
        selected = np.asarray([i for i, particle in enumerate(particles)
                               if particle.code == code], dtype=int)
        if not len(selected):
            continue
        gains = np.stack([particles[i].gains for i in selected]).astype(np.float32)
        state_batch = np.repeat(state[None], len(selected) * len(candidates), axis=0)
        action_batch = np.tile(candidates, (len(selected), 1))
        gain_batch = np.repeat(gains, len(candidates), axis=0)
        with torch.no_grad():
            predicted = filter_model.model(
                torch.as_tensor(state_batch), torch.as_tensor(action_batch), code,
                torch.as_tensor(gain_batch)).cpu().numpy()
        if reward_mode == "velocity_tracking":
            score = -((predicted[:, 6] - target_velocity) ** 2)
        elif reward_mode == "forward_velocity":
            score = np.full(len(predicted), state[6], dtype=float)
        else:
            raise ValueError("unknown reward mode for action portfolio")
        score -= action_cost * np.sum(action_batch ** 2, axis=1)
        particle_scores[selected] = score.reshape(len(selected), len(candidates))
    mean = weights @ particle_scores
    variance = weights @ ((particle_scores - mean) ** 2)
    features = np.concatenate((mean, np.sqrt(np.maximum(variance, 0.0))))
    if (features.shape != (action_portfolio_dimension(filter_model.model.action_dim),)
            or not np.isfinite(features).all()):
        raise ValueError("invalid posterior action-portfolio features")
    return features.astype(np.float32)


class ActionResponseCalibrator(nn.Module):
    """Shared linear correction to the decoder's action-response residual."""

    def __init__(self, decoder: nn.Module, action_center: np.ndarray,
                 response: np.ndarray):
        super().__init__()
        self.decoder = decoder
        self.state_dim = decoder.state_dim
        self.action_dim = decoder.action_dim
        if action_center.shape != (self.action_dim,) or response.shape != (
                self.action_dim, self.state_dim):
            raise ValueError("invalid action-response calibration shape")
        self.register_buffer("action_center", torch.as_tensor(action_center,
                                                               dtype=torch.float32))
        self.register_buffer("response", torch.as_tensor(response,
                                                          dtype=torch.float32))

    @property
    def delta_scale(self) -> torch.Tensor:
        return self.decoder.delta_scale

    def forward(self, states: torch.Tensor, actions: torch.Tensor, code: int,
                gains: torch.Tensor | None = None) -> torch.Tensor:
        return (self.decoder(states, actions, code, gains)
                + (actions - self.action_center) @ self.response)


def fit_action_response(decoder: nn.Module, sequences: Sequence[Batch],
                        *, ridge: float = 1.0) -> ActionResponseCalibrator:
    """Fit a public-data moment correction without simulator counterfactuals."""
    if ridge <= 0 or not sequences:
        raise ValueError("positive ridge and public sequences are required")
    states = np.concatenate([item[0] for item in sequences])
    actions = np.concatenate([item[1] for item in sequences])
    targets = np.concatenate([item[2] for item in sequences])
    if len(states) < 2 or actions.shape != (len(states), decoder.action_dim):
        raise ValueError("invalid public action batches")
    with torch.no_grad():
        x = torch.as_tensor(states, dtype=torch.float32)
        a = torch.as_tensor(actions, dtype=torch.float32)
        pooled = torch.stack([decoder(x, a, code)
                              for code in range(len(SCOPE_PROGRAMS))]).mean(0).numpy()
    center = actions.mean(axis=0)
    centered = actions - center
    response = np.linalg.solve(centered.T @ centered + ridge * np.eye(actions.shape[1]),
                               centered.T @ (targets - pooled))
    return ActionResponseCalibrator(decoder, center, response).eval()


def portfolio_features(filter_model: ScopeLaw, *, predict_next: bool) -> np.ndarray:
    """Fixed-size, public-history-only moments of the weighted law portfolio."""
    particles = filter_model.particles
    weights = np.asarray([p.weight for p in particles], dtype=float)
    weights /= weights.sum()
    codes = np.zeros(len(SCOPE_PROGRAMS), dtype=float)
    evolution = np.zeros(len(filter_model.evolution_laws), dtype=float)
    gains = np.stack([p.gains for p in particles])
    gain_mean = weights @ gains
    gain_var = weights @ ((gains - gain_mean) ** 2)
    switch_probability = 0.0
    age_mean = 0.0
    for p, weight in zip(particles, weights):
        law = filter_model.evolution_laws[p.evolution]
        switch = hazard(p.age, law.scale, law.shape)
        switch_probability += weight * switch
        evolution[p.evolution] += weight
        if predict_next:
            codes[p.code] += weight * (1.0 - switch)
            codes += weight * switch * filter_model.transition[p.code]
            age_mean += weight * (1.0 - switch) * (p.age + 1)
        else:
            codes[p.code] += weight
            age_mean += weight * p.age
    codes /= codes.sum()
    evolution /= evolution.sum()
    code_entropy = -np.sum(codes * np.log(np.maximum(codes, 1e-12))) / log(len(codes))
    evolution_entropy = (-np.sum(evolution * np.log(np.maximum(evolution, 1e-12)))
                         / log(len(evolution)))
    result = np.concatenate((codes, (gain_mean - 1.0) / 0.45,
                             np.sqrt(np.maximum(gain_var, 0.0)) / 0.45,
                             [switch_probability, min(age_mean / 96.0, 1.0),
                              code_entropy, evolution_entropy]))
    if (result.shape != (belief_dimension(filter_model.model.action_dim),)
            or not np.isfinite(result).all()):
        raise ValueError("nonfinite or incorrectly sized SCOPE-Law belief")
    return result.astype(np.float32)


@dataclass(frozen=True)
class PublicObservationEncoder:
    state_mean: np.ndarray
    state_scale: np.ndarray
    use_belief: bool
    use_action_portfolio: bool = False
    belief_dim: int = BELIEF_DIM
    action_portfolio_dim: int = ACTION_PORTFOLIO_DIM

    @classmethod
    def from_states(cls, states: np.ndarray, *, use_belief: bool,
                    use_action_portfolio: bool = False, action_dim: int = 3
                    ) -> "PublicObservationEncoder":
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or len(states) < 2:
            raise ValueError("normalization requires at least two public states")
        return cls(states.mean(axis=0).astype(np.float32),
                   np.maximum(states.std(axis=0), 0.1).astype(np.float32),
                   use_belief, use_action_portfolio,
                   belief_dimension(action_dim), action_portfolio_dimension(action_dim))

    @property
    def dimension(self) -> int:
        return (len(self.state_mean) + self.belief_dim
                + (self.action_portfolio_dim if self.use_action_portfolio else 0))

    def encode(self, state: np.ndarray, belief: np.ndarray | None = None,
               action_portfolio: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != self.state_mean.shape:
            raise ValueError("unexpected public state dimension")
        normalized = np.clip((state - self.state_mean) / self.state_scale, -8.0, 8.0)
        context = (np.asarray(belief, dtype=np.float32) if self.use_belief
                   else np.zeros(self.belief_dim, dtype=np.float32))
        if context.shape != (self.belief_dim,) or not np.isfinite(context).all():
            raise ValueError("invalid causal belief feature")
        result = [normalized, context]
        if self.use_action_portfolio:
            portfolio = (np.zeros(self.action_portfolio_dim, dtype=np.float32)
                         if action_portfolio is None else
                         np.asarray(action_portfolio, dtype=np.float32))
            if (portfolio.shape != (self.action_portfolio_dim,)
                    or not np.isfinite(portfolio).all()):
                raise ValueError("invalid public action-portfolio feature")
            result.append(portfolio)
        elif action_portfolio is not None:
            raise ValueError("encoder was not configured for action-portfolio features")
        return np.concatenate(result).astype(np.float32)


class ReplayBuffer:
    def __init__(self, observation_dim: int, action_dim: int, capacity: int,
                 *, seed: int):
        if capacity < 1:
            raise ValueError("positive replay capacity required")
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.head = 0

    def __len__(self) -> int:
        return self.size

    def add(self, observation: np.ndarray, action: np.ndarray, reward: float,
            next_observation: np.ndarray, terminal: bool) -> None:
        index = self.head
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.terminals[index] = float(terminal)
        self.head = (index + 1) % len(self.rewards)
        self.size = min(self.size + 1, len(self.rewards))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("replay does not yet contain a full batch")
        indices = self.rng.integers(self.size, size=batch_size)
        return tuple(torch.as_tensor(array[indices]) for array in (
            self.observations, self.actions, self.rewards[:, None],
            self.next_observations, self.terminals[:, None]))


def _mlp(input_dim: int, output_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, output_dim))


class _GaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, width: int,
                 action_limit: float, context_dim: int = 0):
        super().__init__()
        self.net = (ZeroStartContextNetwork(observation_dim, context_dim, 2 * action_dim, width)
                    if context_dim else _mlp(observation_dim, 2 * action_dim, width))
        self.action_dim = action_dim
        self.action_limit = float(action_limit)

    def draw(self, observation: torch.Tensor,
             *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(observation).chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 1.0)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else normal.rsample()
        action = self.action_limit * torch.tanh(raw)
        # Stable log-Jacobian of tanh and rescaling to the public action range.
        log_jacobian = 2.0 * (log(2.0) - raw - F.softplus(-2.0 * raw))
        log_probability = (normal.log_prob(raw) - log_jacobian
                           - log(self.action_limit)).sum(dim=-1, keepdim=True)
        return action, log_probability


class _PortfolioGaussianActor(nn.Module):
    """Posterior-weighted law experts with a shared state-only Gaussian prior.

    Each law has a small residual head producing mean/log-scale corrections.
    The posterior over actuator laws mixes these corrections before sampling,
    so uncertainty is preserved as a soft portfolio rather than an argmax law.
    Zero-initialized expert heads make the initial policy exactly the shared
    state-only actor; all experts then learn only from the ordinary SAC loss.
    """

    def __init__(self, observation_dim: int, context_dim: int, action_dim: int,
                 width: int, action_limit: float):
        super().__init__()
        if context_dim < len(SCOPE_PROGRAMS) or observation_dim <= context_dim:
            raise ValueError("portfolio actor requires state and law-probability features")
        self.state_dim = observation_dim - context_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.action_limit = float(action_limit)
        self.base = _mlp(self.state_dim, 2 * action_dim, width)
        self.expert_body = nn.Sequential(nn.Linear(self.state_dim, width), nn.ReLU(),
                                         nn.Linear(width, width), nn.ReLU())
        self.experts = nn.ModuleList(nn.Linear(width, 2 * action_dim)
                                     for _ in SCOPE_PROGRAMS)
        for expert in self.experts:
            nn.init.zeros_(expert.weight)
            nn.init.zeros_(expert.bias)

    def parameters_for(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = observation[..., :self.state_dim]
        probabilities = observation[..., self.state_dim:self.state_dim + len(SCOPE_PROGRAMS)]
        probabilities = probabilities.clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hidden = self.expert_body(state)
        residuals = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        correction = (probabilities.unsqueeze(-1) * residuals).sum(dim=-2)
        mean, log_std = (self.base(state) + correction).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 1.0)

    def draw(self, observation: torch.Tensor,
             *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.parameters_for(observation)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else normal.rsample()
        action = self.action_limit * torch.tanh(raw)
        log_jacobian = 2.0 * (log(2.0) - raw - F.softplus(-2.0 * raw))
        log_probability = (normal.log_prob(raw) - log_jacobian
                           - log(self.action_limit)).sum(dim=-1, keepdim=True)
        return action, log_probability


class _FullBeliefPortfolioGaussianActor(_PortfolioGaussianActor):
    """Law experts conditioned on the full public posterior, including gains.

    The posterior code probabilities still form the soft portfolio weights,
    while each expert receives all belief moments (gain mean/spread, age,
    switch risk, and evolution-law mass) to determine its action correction.
    """

    def __init__(self, observation_dim: int, context_dim: int, action_dim: int,
                 width: int, action_limit: float):
        super().__init__(observation_dim, context_dim, action_dim, width, action_limit)
        del self.expert_body
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(observation_dim, width), nn.ReLU(),
                          nn.Linear(width, width), nn.ReLU(),
                          nn.Linear(width, 2 * action_dim))
            for _ in SCOPE_PROGRAMS)
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def parameters_for(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = observation[..., :self.state_dim]
        probabilities = observation[..., self.state_dim:self.state_dim + len(SCOPE_PROGRAMS)]
        probabilities = probabilities.clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        residuals = torch.stack([expert(observation) for expert in self.experts], dim=-2)
        correction = (probabilities.unsqueeze(-1) * residuals).sum(dim=-2)
        mean, log_std = (self.base(state) + correction).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 1.0)


class _Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, width: int,
                 action_limit: float, context_dim: int = 0):
        super().__init__()
        self.context_dim = context_dim
        self.state_dim = observation_dim-context_dim
        self.net = (ZeroStartContextNetwork(observation_dim+action_dim, context_dim, 1, width)
                    if context_dim else _mlp(observation_dim+action_dim, 1, width))
        self.action_limit = float(action_limit)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.context_dim:
            return self.net(torch.cat((observation[..., :self.state_dim],
                                       action/self.action_limit,
                                       observation[..., self.state_dim:]), dim=-1))
        return self.net(torch.cat((observation, action / self.action_limit), dim=-1))


class SoftActorCritic:
    """Twin-critic maximum-entropy actor-critic with real-reward TD targets."""

    def __init__(self, observation_dim: int, action_dim: int, *, seed: int,
                 action_limit: float = 0.7, width: int = 128, discount: float = 0.99,
                 target_rate: float = 0.01, learning_rate: float = 3e-4,
                 context_dim: int = 0, portfolio_actor: bool | str = False,
                 critic_disagreement_penalty: float = 0.0):
        if not (0 < discount < 1 and 0 < target_rate <= 1 and action_limit > 0):
            raise ValueError("invalid SAC discount, target rate, or action limit")
        if critic_disagreement_penalty < 0 or not np.isfinite(critic_disagreement_penalty):
            raise ValueError("critic disagreement penalty must be finite and nonnegative")
        torch.manual_seed(seed)
        if portfolio_actor == "full_belief":
            self.actor = _FullBeliefPortfolioGaussianActor(
                observation_dim, context_dim, action_dim, width, action_limit)
        elif portfolio_actor:
            self.actor = _PortfolioGaussianActor(observation_dim, context_dim,
                                                 action_dim, width, action_limit)
        else:
            self.actor = _GaussianActor(observation_dim, action_dim, width, action_limit, context_dim)
        self.critics = nn.ModuleList(_Critic(observation_dim, action_dim, width,
                                              action_limit, context_dim) for _ in range(2))
        self.targets = nn.ModuleList(_Critic(observation_dim, action_dim, width,
                                              action_limit, context_dim) for _ in range(2))
        self.targets.load_state_dict(self.critics.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=learning_rate)
        self.log_alpha = nn.Parameter(torch.tensor(log(0.1), dtype=torch.float32))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = -float(action_dim)
        self.discount = discount
        self.target_rate = target_rate
        self.critic_disagreement_penalty = float(critic_disagreement_penalty)

    def act(self, observation: np.ndarray, *, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actor.draw(torch.as_tensor(observation[None],
                                                        dtype=torch.float32),
                                        deterministic=deterministic)
        return action[0].numpy()

    def update(self, replay: ReplayBuffer, batch_size: int = 64) -> dict[str, float]:
        observation, action, reward, next_observation, terminal = replay.sample(batch_size)
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            next_action, next_logp = self.actor.draw(next_observation)
            next_value = torch.minimum(*(target(next_observation, next_action)
                                         for target in self.targets))
            target = reward + self.discount * (1.0 - terminal) * (next_value - alpha * next_logp)
        predictions = [critic(observation, action) for critic in self.critics]
        critic_loss = sum(F.mse_loss(prediction, target) for prediction in predictions)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0)
        self.critic_optimizer.step()

        for parameter in self.critics.parameters():
            parameter.requires_grad_(False)
        chosen_action, logp = self.actor.draw(observation)
        actor_q1, actor_q2 = (critic(observation, chosen_action)
                              for critic in self.critics)
        q = torch.minimum(actor_q1, actor_q2)
        actor_loss = (alpha * logp - q + self.critic_disagreement_penalty
                      * (actor_q1 - actor_q2).abs()).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        for parameter in self.critics.parameters():
            parameter.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for critic, target_critic in zip(self.critics, self.targets):
                for source, target_parameter in zip(critic.parameters(),
                                                    target_critic.parameters()):
                    target_parameter.lerp_(source, self.target_rate)
        metrics = {"critic_loss": float(critic_loss.detach()),
                   "actor_loss": float(actor_loss.detach()),
                   "actor_q_disagreement": float((actor_q1-actor_q2).abs().mean().detach()),
                   "entropy_coefficient": float(self.log_alpha.detach().exp())}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite SAC update")
        return metrics

    def checkpoint(self) -> dict[str, object]:
        return {"actor": self.actor.state_dict(),
                "critics": self.critics.state_dict(),
                "targets": self.targets.state_dict(),
                "log_alpha": self.log_alpha.detach().clone(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict()}
