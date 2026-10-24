import inspect

import numpy as np
import pytest
import torch

from q1.datasets import make_split_manifest
from q1.diagnostics.feasibility_abc import (
    effect_metrics, estimate_old_params, functional_class_ids,
    params_from_vector, sample_idlike_manifest, split_training_background_indices, vector_response,
)
from q1.diagnostics.u4_crossfit import missing_required_files
from q1.direct_effect import DirectEffectMLP, inference_effect
from q1.diagnostics.feasibility_abc import fit_ridge_probe
from q1.simulator import PROGRAMS, DynamicsParams, apply_program
from scripts.q1_feasibility_common import params_for
from scripts.q1_feasibility_c import make_effect_training_target


def _params():
    return DynamicsParams(np.array([[.91,.07],[-.04,.86]]),.36,.112,.23)


def test_a2_estimator_api_only_accepts_old_stage_transitions():
    names=set(inspect.signature(estimate_old_params).parameters)
    assert {"states","actions","delta_v"}.issubset(names)
    assert not any("query" in n or "new_stage" in n or "true_param" in n for n in names)


def test_a2_multistart_recovers_synthetic_low_noise_old_physics_and_obeys_bounds():
    p=_params();rng=np.random.default_rng(801)
    q=rng.uniform(-.7,.7,size=(700,2));v=rng.uniform(-.5,.5,size=(700,2));a=rng.uniform(-1,1,size=(700,2))
    states=np.c_[q,v]
    x=np.c_[states,a]
    y=vector_response(p,x,dt=.05)+rng.normal(0,1e-6,size=(700,2))
    estimated,info=estimate_old_params(states,a,y,dt=.05,seed=54,max_nfev=500,initializations=8)
    np.testing.assert_allclose(estimated.B,p.B,atol=.002,rtol=.002)
    np.testing.assert_allclose([estimated.d,estimated.b,estimated.kappa],[p.d,p.b,p.kappa],atol=.003,rtol=.01)
    assert info["selected_initialization"]<8 and len(info["starts"])==8
    assert info["jacobian_rank"]==7
    assert np.isfinite(info["old_support_rmse"])
    assert np.all(np.abs(estimated.B)<=1.5)
    assert all(.01<=x<=2.0 for x in (estimated.d,estimated.b,estimated.kappa))


def test_program_equations_order_and_class_vs_program_identity():
    p=_params();dt=.05
    x=np.random.default_rng(3).normal(size=(25,6))
    u3=vector_response(apply_program(p,(2,)),x,dt=dt)-vector_response(p,x,dt=dt)
    u4=vector_response(apply_program(p,(3,)),x,dt=dt)-vector_response(p,x,dt=dt)
    v=x[:,2:4];speed=np.linalg.norm(v,axis=1,keepdims=True)
    np.testing.assert_allclose(u3,-.4*dt*p.d*v,atol=1e-15)
    np.testing.assert_allclose(u4,-.5*dt*p.b*speed*v,atol=1e-15)
    assert not np.allclose(apply_program(p,(0,1)).B,apply_program(p,(1,0)).B)
    classes,_=functional_class_ids(p,dt=dt)
    assert PROGRAMS.index((2,3))!=PROGRAMS.index((3,2))
    assert classes[PROGRAMS.index((2,3))]==classes[PROGRAMS.index((3,2))]


def test_b_probe_fit_api_has_no_assessment_labels():
    names=set(inspect.signature(fit_ridge_probe).parameters)
    assert {"train_x","train_y","inner_x","inner_y"}.issubset(names)
    assert not any("test" in n or "assessment" in n or "ood" in n for n in names)
    rng=np.random.default_rng(18);x=rng.normal(size=(30,4));y=np.c_[x[:,0],x[:,1]]
    fitted,meta=fit_ridge_probe(x[:20],y[:20],x[20:],y[20:],[0.,.01,1.])
    assert meta["selection_split"].startswith("background-level inner")
    assert np.isfinite(fitted["coef"]).all()


def test_c_privileged_parameters_are_only_used_by_target_generator_not_inference():
    p=_params();probes=np.random.default_rng(22).normal(size=(10,6))
    target=make_effect_training_target(p,3,probes,dt=.05)
    assert target.shape==(10,2) and np.isfinite(target).all()
    assert {"params","physical_code","probes"}.issubset(inspect.signature(make_effect_training_target).parameters)
    assert {"model","x_norm","context_norm","opaque_code","code_scales"}==set(inspect.signature(inference_effect).parameters)
    assert not any("theta" in n or "params" in n or "program" in n for n in inspect.signature(inference_effect).parameters)
    model=DirectEffectMLP(5,hidden_dim=16,embedding_dim=4)
    out=model(torch.zeros(3,6),torch.zeros(3,5),torch.tensor([0,1,2]))
    assert out.shape==(3,2)


def test_idlike_manifest_avoids_all_saved_seeds_and_forbidden_split_is_unreachable():
    old=make_split_manifest(seed=20261023)
    forbidden=[s for values in old["environment_seeds"].values() for s in values]
    ranges={"train":{"B_diag":[.75,1.0],"B_offdiag":[-.12,.12],"d":[.28,.42],"b":[.08,.14],"kappa":[.18,.28]}}
    new=sample_idlike_manifest(seed=111,count=16,train_ranges=ranges,forbidden_parameter_seeds=forbidden)
    assert len(set(new["parameter_seeds"]))==16
    assert not set(new["parameter_seeds"]) & set(forbidden)
    assert new["regime"]=="IDLIKE_DEVELOPMENT_ONLY"
    with pytest.raises(ValueError):params_for({},"test_id",0,old,{})


def test_fit_inner_and_query_background_indices_are_disjoint():
    fit,inner=split_training_background_indices(count=48,seed=7,fit_count=40,validation_count=8)
    query=set(range(48,64))
    assert len(fit)==40 and len(inner)==8
    assert set(fit).isdisjoint(inner)
    assert set(fit).isdisjoint(query) and set(inner).isdisjoint(query)
    fit2,inner2=split_training_background_indices(count=48,seed=7,fit_count=40,validation_count=8)
    assert (fit,inner)==(fit2,inner2)


def test_physical_delta_v_metrics_and_zero_energy_are_handled_without_epsilon_pass():
    old=np.ones((5,2));new=old.copy();pred=old.copy()
    m=effect_metrics(old,new,pred,np.zeros_like(old))
    assert m["effect_skill"] is None
    assert m["effect_energy_delta_v_squared"]==0.0
    assert m["perfect_effect_identity_abs_error"]<1e-15
    # One physical unit in either component is averaged over both components and rows.
    effect=np.ones((5,2))*.001
    m=effect_metrics(old,old+effect,pred,effect)
    assert m["effect_skill"]==1.0
    assert np.isclose(m["effect_energy_delta_v_squared"],1e-6)


def test_missing_checkpoint_helper_reports_absence_and_never_substitutes(tmp_path):
    good=tmp_path/"local.pt";good.write_bytes(b"saved")
    absent=tmp_path/"missing.pt"
    assert missing_required_files([good,absent])==[str(absent)]


def test_direct_effect_inference_broadcasts_physical_per_code_scale():
    torch.manual_seed(4)
    model=DirectEffectMLP(3,hidden_dim=8,embedding_dim=2).eval()
    x=torch.zeros(5,6);c=torch.zeros(5,3);codes=torch.tensor([0,1,2,3,0])
    pred=inference_effect(model,x,c,codes,torch.tensor([.1,.2,.3,.4]))
    assert pred.shape==(5,2) and torch.isfinite(pred).all()