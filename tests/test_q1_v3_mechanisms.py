from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import torch
from q1_v3.mechanisms import FixedTanhAction,OddActionResponse,mean_delta_v
from q1_v3.operators import MechanismOperatorBank
from q1_v3.program_search import mean_jacobian
from q1_v3.task_inference import fit_task_map
from q1_v3.training import TransitionEpisode,fit_operator_bank,fit_shared_response

def test_odd_action_has_exact_anchor_slope_and_is_odd():
    f=OddActionResponse().double();z=torch.zeros((),dtype=torch.float64,requires_grad=True)
    assert f(z).item()==0.;assert torch.autograd.grad(f(z),z)[0].item()==1.
    x=torch.linspace(-2,2,51,dtype=torch.float64)
    assert torch.allclose(f(-x),-f(x),atol=1e-12)

def test_mechanism_mean_and_analytic_jacobian_match_autodiff():
    torch.manual_seed(2);x=torch.randn(9,6,dtype=torch.float64);theta=torch.tensor([1.,.1,-.2,.8,np.log(.3),np.log(.1),np.log(.2)],dtype=torch.float64,requires_grad=True)
    action=OddActionResponse().double();jac=mean_jacobian(x,theta,action)
    exact=torch.autograd.functional.jacobian(lambda p:mean_delta_v(x,p,action).reshape(-1),theta)
    assert torch.allclose(jac,exact,atol=1e-10,rtol=1e-8)

def test_operator_transform_affine_order_empty_repeat_and_all_groups():
    torch.manual_seed(3);bank=MechanismOperatorBank();theta=torch.randn(7,dtype=torch.float64)
    with torch.no_grad():
        bank.L[0]=torch.tensor([[1.1,.3],[-.1,.9]],dtype=torch.float64)
        bank.L[1]=torch.tensor([[.8,-.2],[.4,1.2]],dtype=torch.float64)
        bank.alpha[0]=torch.tensor([.1,-.2,.3]);bank.alpha[1]=torch.tensor([-.2,.1,.05])
    empty=bank.apply_program(theta,());assert torch.equal(empty,theta)
    a=bank.apply_program(bank.apply_program(theta,(0,)),(1,))
    assert torch.allclose(a,bank.apply_program(theta,(0,1)))
    assert not torch.allclose(a,bank.apply_program(theta,(1,0)))
    assert torch.allclose(bank.apply_program(theta,(0,0))[4:],theta[4:]+2*bank.alpha[0])
    assert bank.affine((0,1))[0].shape==(7,7)

def test_task_map_recovers_in_basis_fixed_tanh_and_returns_finite_covariance():
    torch.manual_seed(4);rng=np.random.default_rng(5);x=torch.tensor(rng.normal(size=(160,6)),dtype=torch.float64)
    true=torch.tensor([.9,.08,-.04,1.05,np.log(.3),np.log(.12),np.log(.22)],dtype=torch.float64)
    action=FixedTanhAction();y=mean_delta_v(x,true,action)+.0001*torch.randn(160,2,dtype=torch.float64)
    mean=true+torch.tensor([.01,-.01,.01,-.01,.03,-.04,.02]);cov=torch.diag(torch.tensor([.1]*4+[.3]*3,dtype=torch.float64))
    fit=fit_task_map(x,y,action,mean,cov,sigma=.0001,starts=3,seed=12,maxiter=100)
    assert fit.converged and torch.isfinite(fit.covariance).all()
    assert torch.linalg.eigvalsh(fit.covariance).min()>0
    assert torch.allclose(fit.theta,true,atol=.01,rtol=.02)

def test_v3_learned_modules_have_no_simulator_oracle_imports():
    root=Path(__file__).parents[1]/'q1_v3'
    for path in root.glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        imports={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        imports|={a.name for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names}
        assert 'q1.simulator' not in imports
        assert not {'apply_program','response','intervention','DynamicsParams'} & {
          a.name for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom)) for a in node.names}

def test_observed_transition_training_smoke_and_complete_balanced_epoch():
    torch.manual_seed(13);action=FixedTanhAction();prior=torch.tensor([1.,0.,0.,1.,np.log(.3),np.log(.1),np.log(.2)],dtype=torch.float64)
    cov=torch.eye(7,dtype=torch.float64)*.2;episodes=[];inner=[]
    for split,dest in [('train',episodes),('inner',inner)]:
      for bg in range(2):
        x=torch.randn(20,6,dtype=torch.float64);theta=prior+torch.randn(7,dtype=torch.float64)*.01
        y=mean_delta_v(x,theta,action)+.0005*torch.randn(20,2,dtype=torch.float64)
        dest.append(TransitionEpisode(x,y,f'{split}-{bg}'))
    fitted,pm,pc,astat=fit_shared_response(episodes,inner,seed=41,epochs=1,dt=.05,sigma=.0005)
    before=pm
    op_train=[];op_inner=[]
    for bg in range(2):
      for code in range(4):
        x=torch.randn(20,6,dtype=torch.float64);y=(mean_delta_v(x,before,fitted)+.0005*torch.randn(20,2,dtype=torch.float64)).detach()
        op_train.append(TransitionEpisode(x,y,f't{bg}',(code,),before))
        op_inner.append(TransitionEpisode(x,y,f'i{bg}',(code,),before))
    bank,meta=fit_operator_bank(fitted,op_train,op_inner,seed=4201,epochs=1,sigma=.0005)
    assert torch.isfinite(bank.L).all() and torch.isfinite(bank.alpha).all()
    assert meta['epoch_coverage'][0]['unique_episode_count']==len(op_train)
    assert set(meta['epoch_coverage'][0]['program_episode_counts'].values())=={2}
    # Repeated seed is bit-for-bit reproducible; a second seed has its own
    # initialization/shuffle stream and must produce a distinct fitted bank.
    same_seed,same_meta=fit_operator_bank(fitted,op_train,op_inner,seed=4201,epochs=1,sigma=.0005)
    other_seed,other_meta=fit_operator_bank(fitted,op_train,op_inner,seed=4202,epochs=1,sigma=.0005)
    assert torch.equal(bank.L,same_seed.L) and torch.equal(bank.alpha,same_seed.alpha)
    assert meta['epoch_coverage'][0]['shuffle_seed']==4201
    assert other_meta['epoch_coverage'][0]['shuffle_seed']==4202
    assert not (torch.equal(bank.L,other_seed.L) and torch.equal(bank.alpha,other_seed.alpha))
    assert torch.isfinite(pc).all() and torch.isfinite(pm).all()
