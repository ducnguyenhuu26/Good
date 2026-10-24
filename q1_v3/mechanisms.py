"""Mechanism-structured velocity-increment model; contains no simulator imports."""
from __future__ import annotations
import math
import torch
from torch import nn

class OddActionResponse(nn.Module):
    """Flexible monotone odd spline with exact zero anchor and unit slope."""
    def __init__(self, hidden: int = 32, num_knots: int = 61, max_input: float = 3.0):
        super().__init__()
        if num_knots<4:raise ValueError('num_knots must be at least 4')
        self.register_buffer('knots',torch.linspace(0.,float(max_input),num_knots,dtype=torch.float64))
        nsegments=num_knots-1
        initial=float(math.log(math.expm1(1.-1e-5)))
        self.raw_slopes=nn.Parameter(torch.full((nsegments-1,),initial,dtype=torch.float64))
        self.min_slope=1e-5
    @property
    def positive_slopes(self):
        variable=torch.nn.functional.softplus(self.raw_slopes)+self.min_slope
        return torch.cat((torch.ones(1,dtype=variable.dtype,device=variable.device),variable))
    def forward(self,t):
        magnitude=t.abs();knots=self.knots.to(dtype=t.dtype,device=t.device)
        slopes=self.positive_slopes.to(dtype=t.dtype,device=t.device)
        idx=torch.searchsorted(knots,magnitude.contiguous(),right=True)-1
        idx=idx.clamp(0,len(knots)-2)
        heights=torch.cat((torch.zeros(1,dtype=t.dtype,device=t.device),torch.cumsum(slopes*(knots[1:]-knots[:-1]),dim=0)))
        value=heights[idx]+(magnitude-knots[idx])*slopes[idx]
        return torch.where(magnitude<knots[1],t,t.sign()*value)
    def regularizer(self,lo:float,hi:float,samples:int=128):
        z=torch.linspace(lo,hi,samples,dtype=next(self.parameters()).dtype,device=next(self.parameters()).device)
        z=z.requires_grad_(True);p=self(z)
        derivative=torch.autograd.grad(p.sum(),z,create_graph=True)[0]
        # Monotonicity is structural; regularize adjacent log slopes on the
        # training-only action range to avoid unsupported sharp bends.
        return .002*torch.diff(torch.log(self.positive_slopes)).square().mean()

class LegacyBoundedOddAction(nn.Module):
    """Frozen reproduction of the original V3 bounded-residual action model."""
    def __init__(self, hidden: int = 8, residual_bound: float = 0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.residual_bound=float(residual_bound)
    def forward(self,t):
        residual=self.residual_bound*torch.tanh(self.net(t.square().unsqueeze(-1)).squeeze(-1))
        return t+t.pow(3)*residual
    def regularizer(self,lo:float,hi:float,samples:int=128):
        z=torch.linspace(lo,hi,samples,dtype=next(self.parameters()).dtype,device=next(self.parameters()).device)
        z=z.requires_grad_(True);p=self(z)
        derivative=torch.autograd.grad(p.sum(),z,create_graph=True)[0]
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
    if not isinstance(action,(OddActionResponse,LegacyBoundedOddAction)):return torch.zeros((),dtype=inputs.dtype,device=inputs.device)
    vals=inputs.detach().abs();limit=float(torch.quantile(vals,0.995).clamp_min(.1))
    return action.regularizer(0.,limit)
