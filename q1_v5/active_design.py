"""Observable-history action selection for the Q1_ACTIVE_EXCITATION protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .identifiability import efficient_information
from .model import DT, SIGMA, jacobian, response


def c_coordinates(c):
    w=np.asarray(c["weights"],dtype=float);s=np.asarray(c["slopes"],dtype=float)
    return np.r_[np.log(w[:2]/w[2]),np.log(s)]


def _from_c(z):
    logits=np.r_[z[:2],0.];w=np.exp(logits-logits.max());w/=w.sum()
    return {"weights":w,"slopes":np.exp(z[2:])}


def full_response_jacobian(x,theta,c,eps=1e-5):
    """Jacobian for seven local task coordinates and five shared-response coordinates."""
    jt=jacobian(np.asarray(x),np.asarray(theta),c);base=c_coordinates(c);jc=[]
    for i in range(len(base)):
        e=np.zeros_like(base);e[i]=eps
        yp=response(np.asarray(x),np.asarray(theta),_from_c(base+e)).reshape(-1)
        ym=response(np.asarray(x),np.asarray(theta),_from_c(base-e)).reshape(-1)
        jc.append((yp-ym)/(2*eps))
    return np.column_stack((jt,np.column_stack(jc)))


def _predict_next(state,action,theta,c,dt):
    x=np.r_[state,action][None,:]
    dv=response(x,theta,c)[0]
    velocity=state[2:]+dv
    return np.r_[state[:2]+dt*velocity,velocity]


def efficient_logb_information(jac,sigma=SIGMA):
    result=efficient_information(jac,5,sigma=sigma)
    return float(result["efficient_information"])


@dataclass
class FisherActionPolicy:
    """Greedy H-step design under a frozen learned response and parameter estimate.

    `efficient_logb` maximizes `log(1 + I_eff(log b) / I_scale)` after adding a
    two-step predicted design to observed history. `d_optimal` is the
    uncertainty-driven comparator and maximizes the log determinant of the
    full twelve-coordinate information matrix with a fixed ridge.
    The same candidate action is held over the short model-predicted horizon.
    """
    objective: str="efficient_logb"
    grid: tuple[float,...]=(-1.,0.,1.)
    horizon: int=2
    dt: float=DT
    sigma: float=SIGMA
    candidates_evaluated: int=0
    last_scores: list[tuple[tuple[float,float],float]]=field(default_factory=list)

    def __post_init__(self):
        if self.objective not in ("efficient_logb","d_optimal"):
            raise ValueError("objective must be efficient_logb or d_optimal")
        if self.horizon<1 or self.dt<=0 or self.sigma<=0:
            raise ValueError("invalid horizon or scale")
        self.actions=np.asarray([(a,b) for a in self.grid for b in self.grid],dtype=float)
        if np.max(np.abs(self.actions))>1+1e-12:
            raise ValueError("candidate actions exceed the simulator bounds")

    def choose(self,state,history_x,theta,c,*,history_information=None):
        """Select from observable state/design history; no labels or simulator object accepted."""
        state=np.asarray(state,dtype=float);history_x=np.asarray(history_x,dtype=float);theta=np.asarray(theta,dtype=float)
        if state.shape!=(4,) or history_x.ndim!=2 or history_x.shape[1]!=6:
            raise ValueError("state/history shapes must be (4,) and (N,6)")
        if history_information is None:
            hist=full_response_jacobian(history_x,theta,c) if len(history_x) else np.zeros((0,12))
            fisher=hist.T@hist/(self.sigma**2)
        else:
            fisher=np.asarray(history_information,dtype=float)
            if fisher.shape!=(12,12) or not np.isfinite(fisher).all():
                raise ValueError("history_information must be finite 12x12")
        scores=[]
        for action in self.actions:
            s=state.copy();future=[]
            for _ in range(self.horizon):
                future.append(np.r_[s,action])
                s=_predict_next(s,action,theta,c,self.dt)
            jf=full_response_jacobian(np.asarray(future),theta,c)
            candidate_fisher=fisher+jf.T@jf/(self.sigma**2)
            if self.objective=="efficient_logb":
                target=5;nuisance=[i for i in range(candidate_fisher.shape[0]) if i!=target]
                a=float(candidate_fisher[target,target]);cross=candidate_fisher[target,nuisance]
                nuisance_fisher=candidate_fisher[np.ix_(nuisance,nuisance)]
                vals,vecs=np.linalg.eigh(.5*(nuisance_fisher+nuisance_fisher.T))
                cutoff=max(nuisance_fisher.shape)*np.finfo(float).eps*max(float(vals.max()),1.)
                keep=vals>cutoff
                inv=(vecs[:,keep]/vals[keep])@vecs[:,keep].T if np.any(keep) else np.zeros_like(nuisance_fisher)
                ieff=max(a-float(cross@inv@cross),0.)
                value=float(np.log1p(ieff))
            else:
                sign,logdet=np.linalg.slogdet(candidate_fisher+np.eye(candidate_fisher.shape[0])*1e-6)
                value=float(logdet) if sign>0 else -float("inf")
            scores.append((tuple(action),value))
        self.last_scores=scores;self.candidates_evaluated+=len(scores)
        # Deterministic stable tie-break by grid order.
        return np.asarray(max(enumerate(scores),key=lambda z:(z[1][1],-z[0]))[1][0],dtype=float)
