"""Observed-transition training for shared response and executable operators."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from .mechanisms import OddActionResponse,mean_delta_v,action_regularizer
from .operators import MechanismOperatorBank
from .task_inference import fit_task_map

@dataclass
class TransitionEpisode:
    x:torch.Tensor
    y:torch.Tensor
    background_id:str
    opaque_program:tuple[int,...]=()
    old_theta:torch.Tensor|None=None

def estimate_prior(theta_rows):
    if isinstance(theta_rows,(list,tuple)):
        t=torch.stack([torch.as_tensor(x,dtype=torch.float64) for x in theta_rows])
    else:t=torch.as_tensor(theta_rows,dtype=torch.float64)
    mean=t.mean(0);centered=t-mean
    covariance=centered.T@centered/max(1,len(t)-1)
    # Floors are train-only and prevent a collapsed empirical prior from removing uncertainty.
    covariance=covariance+torch.diag(torch.tensor([.025**2]*4+[.12**2]*3,dtype=torch.float64))
    return mean,covariance

def _inner_score(action,episodes,prior_mean,prior_cov,*,dt,sigma,epoch_seed):
    scores=[];fits=[]
    for i,ep in enumerate(episodes):
        fit=fit_task_map(ep.x,ep.y,action,prior_mean,prior_cov,dt=dt,sigma=sigma,
            seed=epoch_seed+i,starts=1,maxiter=70)
        fits.append(fit)
        scores.append(fit.nll/max(1,ep.y.numel()))
    return float(np.mean(scores)),fits

def fit_shared_response(train_episodes,inner_episodes,*,seed=20261101,epochs=10,dt=.05,sigma=.0005):
    """Fit one shared odd response and task parameters from noisy observed old laws."""
    torch.manual_seed(int(seed));torch.set_num_threads(1)
    action=OddActionResponse().to(dtype=torch.float64)
    rng=np.random.default_rng(seed)
    init=[]
    for _ in train_episodes:
        init.append(np.array([1.,0.,0.,1.,np.log(.35),np.log(.11),np.log(.22)])+rng.normal(0,[.12,.08,.08,.12,.16,.2,.16]))
    task_parameters=torch.nn.Parameter(torch.as_tensor(np.stack(init),dtype=torch.float64))
    optimizer=torch.optim.Adam([{'params':action.parameters(),'lr':5e-4},{'params':[task_parameters],'lr':.006}])
    x=torch.cat([ep.x for ep in train_episodes]);y=torch.cat([ep.y for ep in train_episodes])
    ids=torch.cat([torch.full((len(ep.x),),i,dtype=torch.long) for i,ep in enumerate(train_episodes)])
    history=[];best=None;best_state=None;best_prior=None;best_inner=[]
    for epoch in range(epochs):
        order=torch.randperm(len(x));losses=[]
        for start in range(0,len(order),4096):
            ix=order[start:start+4096];theta=task_parameters[ids[ix]]
            pred=mean_delta_v(x[ix],theta,action,dt)
            nll=.5*((pred-y[ix])/sigma).square().mean()
            reg=action_regularizer(action,torch.einsum('nij,nj->ni',theta[:,:4].reshape(-1,2,2),x[ix,4:6]))
            loss=nll+.05*reg
            optimizer.zero_grad(set_to_none=True);loss.backward();clip_grad_norm_(list(action.parameters())+[task_parameters],5.)
            optimizer.step();losses.append(float(loss.detach()))
        with torch.no_grad():task_parameters[:,4:7].clamp_(-6.,1.)
        prior_mean,prior_cov=estimate_prior(task_parameters.detach())
        action.eval();val,inner_fits=_inner_score(action,inner_episodes,prior_mean,prior_cov,dt=dt,sigma=sigma,epoch_seed=seed+epoch*100)
        row={'epoch':epoch+1,'train_objective':float(np.mean(losses)),'training_transition_count':len(order),'unique_training_transitions':len(set(order)),'inner_map_nll_per_observation':val,
          'psi_residual_rms':float(action(torch.linspace(-1,1,101,dtype=torch.float64)).sub(torch.linspace(-1,1,101,dtype=torch.float64)).square().mean().sqrt().detach()),
          'inner_map_converged':int(sum(f.converged for f in inner_fits))}
        history.append(row)
        if best is None or val<best:
            best=val;best_state={k:v.detach().clone() for k,v in action.state_dict().items()}
            best_prior=(prior_mean.detach().clone(),prior_cov.detach().clone())
            best_inner=inner_fits
        action.train()
    action.load_state_dict(best_state);action.eval()
    return action,best_prior[0],best_prior[1],{'curve':history,'best_epoch':int(np.argmin([r['inner_map_nll_per_observation'] for r in history])+1),
      'inner_selection_only':True,'inner_map_diagnostics':[{'converged':f.converged,'singular_values':f.jacobian_singular_values,'nll':f.nll} for f in best_inner]}

def fit_operator_bank(action,train_episodes,inner_episodes,*,seed=4201,dt=.05,sigma=.0005,
  epochs=5,lr=.003,sparsity=2e-4,stability=1e-4):
    """Train all four uniform operators using observed after-stage delta-v only."""
    torch.manual_seed(int(seed));bank=MechanismOperatorBank(dtype=torch.float64)
    optimizer=torch.optim.Adam(bank.parameters(),lr=lr);history=[];coverage=[];best=float('inf');best_state=None
    for epoch in range(epochs):
        rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(train_episodes));train_losses=[]
        programs={str(ep.opaque_program):0 for ep in train_episodes}
        for i in order:programs[str(train_episodes[i].opaque_program)]+=1
        coverage.append({'epoch':epoch+1,'episode_count':len(order),'unique_episode_count':len(set(order)),
          'program_episode_counts':programs,'shuffle_seed':int(seed+epoch)})
        for start in range(0,len(order),8):
            batch=[train_episodes[i] for i in order[start:start+8]]
            losses=[]
            for ep in batch:
                theta=bank.apply_program(ep.old_theta,ep.opaque_program)
                prediction=mean_delta_v(ep.x,theta,action,dt)
                losses.append(.5*((prediction-ep.y)/sigma).square().mean())
            nll=torch.stack(losses).mean()
            penalty=sparsity*bank.sparsity_penalty()+stability*((bank.L-torch.eye(2,dtype=torch.float64)).square().mean()+bank.alpha.square().mean())
            loss=nll+penalty
            optimizer.zero_grad(set_to_none=True);loss.backward();clip_grad_norm_(bank.parameters(),5.);optimizer.step()
            train_losses.append(float(nll.detach()))
        val=operator_validation_nll(action,bank,inner_episodes,dt=dt,sigma=sigma)
        history.append({'epoch':epoch+1,'train_nll_per_observation':float(np.mean(train_losses)),'inner_nll_per_observation':val,
          'group_sparsity_penalty':float(bank.sparsity_penalty().detach()),'selected':val<best})
        if val<best:
            best=val;best_state={k:v.detach().clone() for k,v in bank.state_dict().items()}
    bank.load_state_dict(best_state)
    return bank,{'curve':history,'epoch_coverage':coverage,'best_epoch':int(np.argmin([r['inner_nll_per_observation'] for r in history])+1),
      'inner_selection_only':True,'best_inner_nll_per_observation':best}

def operator_validation_nll(action,bank,episodes,*,dt,sigma):
    vals=[]
    with torch.no_grad():
        for ep in episodes:
            pred=mean_delta_v(ep.x,bank.apply_program(ep.old_theta,ep.opaque_program),action,dt)
            vals.append(.5*((pred-ep.y)/sigma).square().mean().item())
    return float(np.mean(vals))
