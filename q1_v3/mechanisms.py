"""Mechanism-structured velocity-increment model; contains no simulator imports."""
from __future__ import annotations
import math
import torch
from torch import nn

class OddActionResponse(nn.Module):
    """Odd per-coordinate response with exact zero anchor and unit local slope."""
    def __init__(self, hidden: int = 8, residual_bound: float = 0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.residual_bound=float(residual_bound)
    def forward(self,t):
        s=t.square().unsqueeze(-1)
        residual=self.residual_bound*torch.tanh(self.net(s).squeeze(-1))
        return t+t.pow(3)*residual
    def regularizer(self,lo:float,hi:float,samples:int=128):
        z=torch.linspace(lo,hi,samples,dtype=next(self.parameters()).dtype,device=next(self.parameters()).device)
        z=z.requires_grad_(True);p=self(z)
        derivative=torch.autograd.grad(p.sum(),z,create_graph=True)[0]
        # Bounded-range monotonicity plus a weak curvature/smoothness penalty.
        return torch.relu(.05-derivative).square().mean()+.002*torch.diff(derivative).square().mean()

class FixedTanhAction(nn.Module):
    """Simulator-matched, privileged diagnostic reference; not the learned model."""
    def forward(self,t): return torch.tanh(t)
    def regularizer(self,*args,**kwargs): return torch.zeros((),dtype=torch.float64)

def unpack_theta(theta):
    """theta uses row-major B[4], then log(d), log(b), log(kappa)."""
    return theta[..., :4].reshape(*theta.shape[:-1],2,2),theta[...,4:7].exp()

def mean_delta_v(x,theta,action,dt:float=.05):
    """Conditional mean delta-v; x is (..., qx,qy,vx,vy,ax,ay)."""
    B,rates=unpack_theta(theta)
    q=x[...,0:2];v=x[...,2:4];a=x[...,4:6]
    if B.ndim==2:
        input_action=torch.einsum('ij,...j->...i',B,a)
    else:
        input_action=torch.einsum('...ij,...j->...i',B,a)
    force=(action(input_action)-rates[...,0,None]*v
      -rates[...,1,None]*torch.linalg.vector_norm(v,dim=-1,keepdim=True)*v
      -rates[...,2,None]*q)
    return dt*force

def action_regularizer(action,inputs):
    if not isinstance(action,OddActionResponse):return torch.zeros((),dtype=inputs.dtype,device=inputs.device)
    vals=inputs.detach().abs();limit=float(torch.quantile(vals,0.995).clamp_min(.1))
    return action.regularizer(0.,limit)
