"""Online predictive-mismatch detection, residual theory revision and memory.

The shared predictor is frozen. Online learning changes only a small RLS
residual adapter (10 scalar weights for 2D velocity residuals). No environment
labels, switch times, or simulator parameters enter the agent-facing methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rwt.stage2b import PredictionSet


def residual_features(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """A compact continuous feature map for the velocity correction."""
    s = np.asarray(state, dtype=np.float64)
    a = np.asarray(action, dtype=np.float64)
    if s.shape != (4,) or a.shape != (2,):
        raise ValueError("expected state (4,) and action (2,)")
    return np.concatenate(([1.0], s[2:], a))


@dataclass
class RLSResidual:
    """Recursive least squares adapter: 5 inputs x 2 velocity outputs."""
    ridge: float = 8.0
    forgetting: float = 1.0
    weights: np.ndarray = field(default_factory=lambda: np.zeros((5, 2), dtype=np.float64))
    covariance: np.ndarray = field(init=False, repr=False)
    updates: int = 0

    def __post_init__(self):
        if self.ridge <= 0 or not 0 < self.forgetting <= 1:
            raise ValueError("invalid RLS parameters")
        self.covariance = np.eye(5, dtype=np.float64) / self.ridge

    @property
    def parameter_count(self) -> int:
        return int(self.weights.size)

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return residual_features(state, action) @ self.weights

    def predictive_variance(self, state: np.ndarray, action: np.ndarray, scale: float) -> float:
        phi=residual_features(state,action)
        return float(max(0.0,phi@self.covariance@phi)*scale*scale)

    def update(self, state: np.ndarray, action: np.ndarray, residual: np.ndarray) -> None:
        phi = residual_features(state, action)
        y = np.asarray(residual, dtype=np.float64)
        if y.shape != (2,) or not np.isfinite(y).all():
            raise ValueError("residual must be a finite 2-vector")
        p_phi = self.covariance @ phi
        denom = self.forgetting + float(phi @ p_phi)
        gain = p_phi / denom
        error = y - phi @ self.weights
        self.weights += np.outer(gain, error)
        self.covariance = (self.covariance - np.outer(gain, phi @ self.covariance)) / self.forgetting
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.updates += 1

    def copy(self) -> "RLSResidual":
        out = RLSResidual(self.ridge, self.forgetting)
        out.weights = self.weights.copy()
        out.covariance = self.covariance.copy()
        out.updates = self.updates
        return out


@dataclass
class StoredTheory:
    """An observed residual revision plus its predictive signature."""
    anchor_index: int
    adapter: RLSResidual
    signature: np.ndarray
    observations: int
    retrievals: int = 0


@dataclass
class RevisionStep:
    mean_velocity: np.ndarray
    nll: float
    mse: float
    map_index: int
    behavioral_accuracy: float
    mismatch_score: float
    mismatch_alarm: bool
    created: bool
    cache_hit: bool
    active_updates: int
    memory_size: int
    epistemic_variance: float
    aleatoric_variance: float
    selected_index: int = -1


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


class OnlineTheoryRevision:
    """Bayesian predictive bank with delayed residual creation and retrieval.

    ``predict`` is called before each environment transition. ``observe`` is
    called only after its next state is available. The detector requires at
    least two calibrated surprises in a rolling three-transition window.
    """
    def __init__(self, prediction_factory, names: tuple[str, ...], *,
                 completion_indices: tuple[int, ...], mismatch_threshold: float,
                 memory: bool = True, prior_scale: float = 0.03, hazard: float = 0.05):
        if not names or mismatch_threshold <= 0 or prior_scale <= 0:
            raise ValueError("invalid revision bank configuration")
        self.prediction_factory = prediction_factory
        self.names = tuple(names)
        self.completion_indices = tuple(completion_indices)
        self.threshold = float(mismatch_threshold)
        self.memory_enabled = bool(memory)
        self.prior_scale = float(prior_scale)
        self.hazard = float(hazard)
        if not 0 <= self.hazard <= 1:
            raise ValueError("hazard must lie in [0,1]")
        self.posterior = np.full(len(names), 1 / len(names), dtype=np.float64)
        self.memory: list[StoredTheory] = []
        self.active: StoredTheory | None = None
        self.alarm_history: list[bool] = []
        self.map_history: list[int] = []
        self.last_predictions: PredictionSet | None = None
        self.last_candidates: list[tuple[int, RLSResidual | None, StoredTheory | None]] = []
        self.pending = None
        self.last_prior = self.posterior.copy()
        self.last_mismatch_score = 0.0
        self.revision_count = 0
        self.cache_hits = 0
        self.false_retrievals = 0
        self.current_cache_index: int | None = None

    def _candidate_bank(self, state: np.ndarray, action: np.ndarray):
        base = self.prediction_factory(state, action)
        candidates = [(i, None, None) for i in range(len(base.names))]
        means = [x.copy() for x in base.means]
        scales = [x.copy() for x in base.scales]
        if self.active is not None:
            i = self.active.anchor_index
            means.append(base.means[i] + self.active.adapter.predict(state, action))
            scales.append(np.sqrt(base.scales[i] ** 2 + self.active.adapter.predictive_variance(state,action,self.prior_scale)))
            candidates.append((i, self.active.adapter, self.active))
        if self.memory_enabled:
            for record in self.memory:
                i = record.anchor_index
                means.append(base.means[i] + record.adapter.predict(state, action))
                scales.append(np.sqrt(base.scales[i] ** 2 + record.adapter.predictive_variance(state,action,self.prior_scale)))
                candidates.append((i, record.adapter, record))
        return base, np.asarray(means), np.asarray(scales), candidates

    def predict(self, state: np.ndarray, action: np.ndarray):
        base, means, scales, candidates = self._candidate_bank(state, action)
        n = len(base.names)
        prior = np.full(len(candidates), 1e-3 / max(1, len(candidates) - n), dtype=np.float64)
        base_weights = (1.0 - self.hazard) * self.posterior + self.hazard / n
        prior[:n] = base_weights * 0.94
        if self.active is not None:
            prior[n] = max(prior[n], 0.02)
        if len(candidates) > n + int(self.active is not None):
            # Persistent theories are eligible for retrieval on every step.
            tail = n + int(self.active is not None)
            prior[tail:] = 0.04 / (len(candidates) - tail)
        prior /= prior.sum()
        mixture = np.einsum("k,kd->d", prior, means)
        self.last_predictions = PredictionSet(
            names=tuple([*base.names, *(f"revision_{i}" for i in range(len(candidates)-n))]),
            means=means, scales=scales, known_count=base.known_count,
            completion_indices=tuple(i for i in base.completion_indices),
        ) if len(candidates) == len(base.names) else None
        self.last_candidates = candidates
        self.last_prior = prior
        self.pending = (base, means, scales, candidates, prior, mixture)
        return self.pending

    def observe(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> RevisionStep:
        if self.pending is None:
            raise RuntimeError("predict must be called before observe")
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)
        base, means, scales, candidates, prior, mixture = self.pending
        self.pending = None
        observation = next_state[2:]
        z = (observation[None, :] - means) / scales
        ll = -0.5 * (z ** 2 + 2 * np.log(scales) + np.log(2 * np.pi)).sum(axis=-1)
        nll = -_logsumexp(np.log(prior.clip(1e-300)) + ll)
        predicted_state = np.concatenate((state[:2] + mixture, mixture))
        mse = float(np.mean((predicted_state - next_state) ** 2))
        # Behavioral correctness is whether the posterior MAP predictor is
        # also the candidate with the smallest realized squared residual.
        posterior_log = np.log(prior.clip(1e-300)) + ll
        posterior_log -= _logsumexp(posterior_log)
        posterior = np.exp(posterior_log)
        map_idx = int(np.argmax(posterior))
        mse_by_candidate = np.mean((means - observation[None, :]) ** 2, axis=1)
        behavioral = float(map_idx == int(np.argmin(mse_by_candidate)))
        base_n = len(base.names)
        # Best seen/completion candidate mismatch score, normalized by its
        # predictive standard deviation. This separates model uncertainty
        # (variance across candidate means) from irreducible predicted noise.
        all_z = np.sqrt(np.mean(((observation[None, :] - means) / scales) ** 2, axis=-1))
        mismatch = float(np.min(all_z))
        alarm = mismatch > self.threshold
        self.alarm_history.append(alarm)
        self.alarm_history = self.alarm_history[-3:]
        created = False
        # Scores after observing this transition may trigger a revision, but
        # that transition is then used only for the post-update RLS step.
        self.posterior = posterior[:base_n].copy()
        if self.posterior.sum() > 0:
            self.posterior /= self.posterior.sum()
        if self.active is None and len(self.alarm_history) >= 3 and sum(self.alarm_history) >= 2 and self.completion_indices:
            # Select an anchor from candidate likelihood accumulated so far,
            # never from the evaluator regime label.
            anchor = max(self.completion_indices, key=lambda i: ll[i])
            self.active = StoredTheory(anchor, RLSResidual(), np.zeros(10), 0)
            self.revision_count += 1
            created = True
        # Select/update the active or retrieved theory only after this observed
        # transition; no unobserved/future transitions are buffered.
        record = self.active if created else (candidates[map_idx][2] if map_idx >= base_n else None)
        cache_hit = record is not None and any(record is item for item in self.memory)
        if cache_hit:
            record.retrievals += 1
            record_index = next(i for i, item in enumerate(self.memory) if record is item)
            self.cache_hits += int(self.current_cache_index != record_index)
            self.current_cache_index = record_index
        elif self.current_cache_index is not None:
            self.current_cache_index = None
        updated = 0
        if record is not None:
            anchor_mean = base.means[record.anchor_index]
            residual = observation - anchor_mean
            record.adapter.update(state, action, residual)
            record.observations += 1
            updated = 1
            # Store a predictive residual signature independent of regime IDs.
            record.signature = record.adapter.weights.reshape(-1).copy()
        # A confident known-theory switch is an online boundary cue. Archive
        # the revised model only after it has accumulated several observations.
        if self.active is not None and self.memory_enabled and self.active.observations >= 4:
            known_mass = float(self.posterior[:base.known_count].sum())
            base_map = int(np.argmax(self.posterior))
            if self.posterior[base_map] >= 0.9 and base_map != self.active.anchor_index:
                if all(record.signature.shape != self.active.signature.shape or
                       not np.allclose(record.signature, self.active.signature) for record in self.memory):
                    self.memory.append(self.active)
                self.active = None
        self.last_mismatch_score = mismatch
        # Total predictive variance decomposition over the prior candidate bank.
        ep = float(np.mean(np.sum(prior[:, None] * (means - mixture) ** 2, axis=0)))
        al = float(np.mean(np.sum(prior[:, None] * scales ** 2, axis=0)))
        return RevisionStep(mixture, float(nll), mse, map_idx, behavioral, mismatch,
                            alarm, created, cache_hit, updated,
                            len(self.memory), ep, al)


class StreamingTheoryAdapters:
    """Online residual adaptation of candidate theories without GT labels.

    This complementary ablation updates the transition-best theory after each
    observed sample. With ``memory=True`` each theory owns a separate sparse
    RLS adapter; without memory one adapter is discarded when the selected
    theory changes. The original learned predictors remain frozen.
    """
    def __init__(self, prediction_factory, names: tuple[str, ...], *,
                 memory: bool, hazard: float = .05, ridge: float = 8.0,
                 adapter_scale: float = .03):
        self.prediction_factory = prediction_factory
        self.names = tuple(names); self.memory_enabled = bool(memory)
        self.hazard = float(hazard); self.ridge = float(ridge); self.adapter_scale = float(adapter_scale)
        if not names or not 0 <= hazard <= 1 or ridge <= 0 or adapter_scale <= 0:
            raise ValueError("invalid streaming adapter settings")
        self.posterior = np.full(len(names),1/len(names),dtype=np.float64)
        self.adapters: dict[int,RLSResidual] = {}
        self.signatures: dict[int,np.ndarray] = {}
        self.active_index: int | None = None
        self.active_adapter: RLSResidual | None = None
        self.pending = None
        self.total_updates = 0

    def predict(self,state: np.ndarray,action: np.ndarray):
        base=self.prediction_factory(state,action)
        means=base.means.copy(); scales=base.scales.copy(); adapter_variances=np.zeros(len(self.names))
        if self.memory_enabled:
            for i,adapter in self.adapters.items():
                means[i]+=adapter.predict(state,action)
                adapter_variances[i]=adapter.predictive_variance(state,action,self.adapter_scale)
        elif self.active_index is not None and self.active_adapter is not None:
            means[self.active_index]+=self.active_adapter.predict(state,action)
            adapter_variances[self.active_index]=self.active_adapter.predictive_variance(state,action,self.adapter_scale)
        scales=np.sqrt(scales**2+adapter_variances[:,None])
        prior=(1-self.hazard)*self.posterior+self.hazard/len(self.names)
        mixture=np.einsum('k,kd->d',prior,means)
        self.pending=(base,means,scales,prior,mixture,state.copy(),action.copy())
        return self.pending

    def observe(self,next_state: np.ndarray):
        if self.pending is None: raise RuntimeError("predict must precede observe")
        base,means,scales,prior,mixture,state,action=self.pending; self.pending=None
        observed=np.asarray(next_state,dtype=np.float64)[2:]
        z=(observed[None]-means)/scales
        ll=-.5*(z*z+2*np.log(scales)+np.log(2*np.pi)).sum(-1)
        logpost=np.log(prior.clip(1e-300))+ll; logpost-=_logsumexp(logpost)
        post=np.exp(logpost); map_idx=int(np.argmax(post))
        fit=np.mean((means-observed[None])**2,axis=-1)
        predicted_state=np.r_[state[:2]+mixture,mixture]
        result_mse=float(np.mean((predicted_state-np.asarray(next_state))**2))
        nll=float(-_logsumexp(np.log(prior.clip(1e-300))+ll))
        cache_hit=False
        selected=int(np.argmax(ll))
        if self.memory_enabled:
            adapter=self.adapters.setdefault(selected,RLSResidual(ridge=self.ridge))
            cache_hit=adapter.updates>0
        else:
            if self.active_index!=selected:
                self.active_index=selected; self.active_adapter=RLSResidual(ridge=self.ridge)
            adapter=self.active_adapter
        adapter.update(state,action,observed-base.means[selected])
        self.signatures[selected]=adapter.weights.reshape(-1).copy()
        self.total_updates+=1; self.posterior=post
        mismatch=float(np.min(np.sqrt(np.mean(((observed[None]-base.means)/base.scales)**2,axis=-1))))
        adapter_variances=np.mean(np.maximum(scales**2-base.scales**2,0),axis=1)
        ep=float(np.mean(np.sum(prior[:,None]*(means-mixture)**2,axis=0))+np.sum(prior*adapter_variances))
        al=float(np.mean(np.sum(prior[:,None]*base.scales**2,axis=0)))
        return RevisionStep(mixture,nll,result_mse,map_idx,float(map_idx==int(np.argmin(fit))),
                            mismatch,False,False,cache_hit,1,len(self.adapters),ep,al,selected)


__all__ = ["OnlineTheoryRevision", "RLSResidual", "RevisionStep", "StoredTheory",
           "StreamingTheoryAdapters", "residual_features"]
