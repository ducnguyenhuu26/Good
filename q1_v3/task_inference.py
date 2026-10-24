"""Bounded, prior-regularized seven-coordinate MAP and Laplace uncertainty."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.optimize import minimize
from .mechanisms import mean_delta_v

@dataclass
class MAPResult:
    theta: torch.Tensor
    covariance: torch.Tensor
    nll: float
    converged: bool
    starts: int
    jacobian_singular_values: list[float]
    iterations: int

def fit_task_map(x,y,action,prior_mean,prior_cov,*,dt=.05,sigma=.0005,seed=0,starts=3,maxiter=150):
    x=torch.as_tensor(x,dtype=torch.float64);y=torch.as_tensor(y,dtype=torch.float64).reshape(-1,2)
    mean=torch.as_tensor(prior_mean,dtype=torch.float64);cov=torch.as_tensor(prior_cov,dtype=torch.float64)
    precision=torch.linalg.inv(cov+1e-8*torch.eye(7,dtype=torch.float64))
    chol=torch.linalg.cholesky(cov+1e-8*torch.eye(7,dtype=torch.float64))
    bounds=[(-2.5,2.5)]*4+[(-6.0,1.0)]*3
    rng=np.random.default_rng(seed)
    initial=[mean.detach().numpy()]
    for i in range(starts-1):
        z=np.zeros(7);z[:4]=rng.normal(0,.18,4);z[4:]=rng.normal(0,.25,3)
        initial.append((mean+chol@torch.as_tensor(z,dtype=torch.float64)).detach().numpy())
    def objective(z):
        t=torch.tensor(z,dtype=torch.float64,requires_grad=True)
        residual=mean_delta_v(x,t,action,dt)-y
        value=.5*residual.square().sum()/(sigma*sigma)+.5*(t-mean)@precision@(t-mean)
        grad=torch.autograd.grad(value,t)[0]
        return float(value.detach()),grad.detach().numpy().astype(np.float64)
    candidates=[]
    for start in initial:
        r=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=bounds,
          options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-6,'maxls':30})
        candidates.append(r)
    best=min(candidates,key=lambda r:float(r.fun));theta=torch.tensor(best.x,dtype=torch.float64)
    t=theta.detach().requires_grad_(True)
    J=torch.autograd.functional.jacobian(lambda p:mean_delta_v(x,p,action,dt),t).reshape(-1,7)
    singular=torch.linalg.svdvals(J)
    hessian=J.T@J/(sigma*sigma)+precision
    hessian=.5*(hessian+hessian.T)+1e-8*torch.eye(7,dtype=torch.float64)
    covariance=torch.linalg.inv(hessian)
    finite=bool(torch.isfinite(theta).all() and torch.isfinite(covariance).all() and float(torch.linalg.eigvalsh(covariance).min())>0)
    return MAPResult(theta,covariance,float(best.fun),bool(best.success and finite),starts,
      [float(v) for v in singular],int(best.nit))
