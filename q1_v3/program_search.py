"""Candidate program composition, point-MAP and linearized Laplace scoring."""
from __future__ import annotations
import math
import torch
from .mechanisms import mean_delta_v

def mean_jacobian(x,theta,action,dt=.05):
    """Analytic seven-coordinate Jacobian, with autodiff only for learned psi'."""
    x=x.detach();theta=theta.detach()
    B=theta[:4].reshape(2,2);rates=theta[4:7].exp();a=x[:,4:6];q=x[:,:2];v=x[:,2:4]
    u=a@B.T;u=u.detach().requires_grad_(True);p=action(u)
    dpsi=[]
    for k in range(2):dpsi.append(torch.autograd.grad(p[:,k].sum(),u,retain_graph=(k==0))[0][:,k])
    dpsi=torch.stack(dpsi,dim=1)
    pieces=[]
    for out in range(2):
        for row in range(2):
            for col in range(2):pieces.append(dt*dpsi[:,out]*a[:,col] if row==out else torch.zeros_like(a[:,0]))
        pieces.append(-dt*rates[0]*v[:,out]);pieces.append(-dt*rates[1]*torch.linalg.vector_norm(v,dim=-1)*v[:,out]);pieces.append(-dt*rates[2]*q[:,out])
    # Construct row-major output rows: [output0 params0:7, output1 params0:7] per row.
    cols=[]
    for out in range(2):
        local=[]
        for row in range(2):
            for col in range(2):local.append(dt*dpsi[:,out]*a[:,col] if row==out else torch.zeros_like(a[:,0]))
        local += [-dt*rates[0]*v[:,out],-dt*rates[1]*torch.linalg.vector_norm(v,dim=-1)*v[:,out],-dt*rates[2]*q[:,out]]
        cols.extend(local)
    # Interleave x/y output Jacobian rows to match flatten(delta-v).
    return torch.stack(cols,dim=-1).reshape(len(x),2,7).reshape(-1,7)

def marginal_nll(x,y,theta,covariance,action,*,dt=.05,sigma=.0005):
    """Woodbury NLL; covariance is propagated in 7D and predictive mean is linearized."""
    theta=theta.detach().requires_grad_(True)
    pred=mean_delta_v(x,theta,action,dt).reshape(-1)
    J=torch.autograd.functional.jacobian(lambda p:mean_delta_v(x,p,action,dt).reshape(-1),theta).detach()
    residual=(y.reshape(-1)-pred.detach())
    S=.5*(covariance+covariance.T)+1e-10*torch.eye(7,dtype=torch.float64)
    C=torch.linalg.cholesky(S)
    U=J@C
    K=torch.eye(7,dtype=torch.float64)+U.T@U/(sigma*sigma)
    chol=torch.linalg.cholesky(K)
    z=U.T@residual/(sigma*sigma)
    solved=torch.cholesky_solve(z[:,None],chol)[:,0]
    quad=residual.square().sum()/(sigma*sigma)-z@solved
    logdet=2*torch.log(torch.diag(chol)).sum()+residual.numel()*math.log(sigma*sigma)
    return float(.5*(quad+logdet+residual.numel()*math.log(2*math.pi)).detach())

def score_programs(x,y,old_theta,old_cov,bank,action,programs,*,dt=.05,sigma=.0005,length_penalty=0.):
    rows=[]
    for program in programs:
        M,c=bank.affine(program);theta=M@old_theta+c;cov=M@old_cov@M.T
        with torch.no_grad():pred=mean_delta_v(x,theta,action,dt).reshape(-1)
        point=float(.5*((pred-y.reshape(-1))/sigma).square().sum()+length_penalty*len(program))
        marginal=marginal_nll(x,y,theta,cov,action,dt=dt,sigma=sigma)+length_penalty*len(program)
        rows.append({'program':tuple(program),'point_score':point,'marginal_score':marginal,
          'query_delta_v_mse':float((pred-y.reshape(-1)).square().mean()),'theta_mean':theta.detach(),
          'theta_covariance':cov.detach(),'predicted_mean':pred.detach()})
    return rows

def score_program_budgets(x,y,old_theta,old_cov,bank,action,programs,budgets,*,dt=.05,sigma=.0005,length_penalty=0.):
    """Reuse each program Jacobian across prefixes of one uninterrupted adaptation."""
    out={int(n):[] for n in budgets}
    for program in programs:
        M,c=bank.affine(program);theta=M@old_theta+c;cov=M@old_cov@M.T
        with torch.no_grad():pred=mean_delta_v(x,theta,action,dt).reshape(-1)
        J=mean_jacobian(x,theta,action,dt);C=torch.linalg.cholesky(.5*(cov+cov.T)+1e-10*torch.eye(7,dtype=torch.float64));U=J@C
        for n in budgets:
            count=2*int(n);res=(y[:n].reshape(-1)-pred[:count]);u=U[:count]
            K=torch.eye(7,dtype=torch.float64)+u.T@u/(sigma*sigma);chol=torch.linalg.cholesky(K)
            z=u.T@res/(sigma*sigma);quad=res.square().sum()/(sigma*sigma)-z@torch.cholesky_solve(z[:,None],chol)[:,0]
            logdet=2*torch.log(torch.diag(chol)).sum()+count*math.log(sigma*sigma)
            marginal=float(.5*(quad+logdet+count*math.log(2*math.pi)).detach())+length_penalty*len(program)
            point=float(.5*(res/sigma).square().sum())+length_penalty*len(program)
            out[int(n)].append({'program':tuple(program),'point_score':point,'marginal_score':marginal,
              'query_delta_v_mse':float(res.square().mean()),'theta_mean':theta.detach(),'theta_covariance':cov.detach()})
    return out

def predict_effect(x,theta,bank,code,action,*,dt=.05):
    with torch.no_grad():
        return (mean_delta_v(x,bank.apply_program(theta,(code,)),action,dt)-mean_delta_v(x,theta,action,dt))
