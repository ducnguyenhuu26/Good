import numpy as np
import torch
import pytest

from q1.datasets import make_split_manifest, sample_environment, stage_pair
from q1.diagnostics.u4_crossfit import (
    _svd_ridge_solve,
    batched_context_jacobian,
    blocked_inner_indices,
    effect_and_prediction_metrics,
    ensure_write_once_directory,
    missing_required_files,
    predict_crossfit_query,
    select_fit_ridge_and_refit,
    solve_query_refit_upper_bound,
)
from q1.model import ContextDynamicsModel, TransitionNormalizer
from q1.primitives import map_intervention_program
from q1.simulator import apply_program, response


def _normalizer():
    return TransitionNormalizer(np.array([.2,-.1,.1,0.,0.,0.]),
        np.array([.5,.7,.8,.6,1.2,1.1]), np.array([.1,-.2]), np.array([.003,.005]))


def test_perfect_effect_identity_and_arbitrary_decomposition():
    rng=np.random.default_rng(3)
    f0=rng.normal(size=(20,2)); delta=rng.normal(scale=.01,size=(20,2)); f1=f0+delta
    fhat0=f0+rng.normal(size=(20,2)); fhat1=fhat0+rng.normal(size=(20,2))
    got=effect_and_prediction_metrics(pred_old=fhat0,pred_new=fhat1,true_old=f0,true_new=f1)
    np.testing.assert_allclose(got["perfect_effect_after_mse_delta_v_squared"],
        got["base_error_delta_v_squared"],rtol=1e-13,atol=1e-14)
    assert got["decomposition_identity_pass"]
    assert got["decomposition_max_abs_error_delta_v"]<1e-14


def test_bias_confound_perfect_effect_does_not_imply_five_percent_after_gain():
    f0=np.zeros((16,2)); delta=np.full((16,2),.01); f1=f0+delta
    fhat0=np.full((16,2),-1.0)
    metrics=effect_and_prediction_metrics(pred_old=fhat0,pred_new=fhat0+delta,true_old=f0,true_new=f1)
    assert metrics["effect_skill"]==1.0
    assert metrics["perfect_effect_gain"]<.05
    assert metrics["learned_after_gain"]<.05
    assert metrics["perfect_effect_identity_abs_error"]<1e-14


def test_simulator_perfect_effect_identity_uses_paired_conditional_means():
    p=sample_environment(make_split_manifest(),"validation",0)
    traj=stage_pair(p,(3,),seed=4401,n_support=96,n_query=80,dt=.05,sigma=.0005)
    query=traj[2]
    raw=np.column_stack((query.states[:-1],query.actions))
    f0=response(p,raw,dt=.05); f1=response(apply_program(p,(3,)),raw,dt=.05)
    rng=np.random.default_rng(5)
    pred_old=f0+rng.normal(scale=.004,size=f0.shape)
    arbitrary_new=pred_old+rng.normal(scale=.002,size=f0.shape)
    metrics=effect_and_prediction_metrics(pred_old=pred_old,pred_new=arbitrary_new,true_old=f0,true_new=f1)
    assert metrics["perfect_effect_identity_abs_error"]<1e-15
    assert metrics["decomposition_identity_pass"]


def test_crossfit_selection_is_independent_of_query_effect_labels():
    rng=np.random.default_rng(20)
    jfit=rng.normal(size=(60,4)); dtrue=np.array([.2,-.1,.05,.4])
    yfit=jfit@dtrue
    train=np.arange(40); valid=np.arange(48,60)
    jquery=rng.normal(size=(80,4)); true_q=jquery@dtrue
    dc1,diag1=select_fit_ridge_and_refit(jfit,yfit,jfit[train],yfit[train],
        jfit[valid],yfit[valid],ridge_factors=[0.,1e-6,1e-3])
    predictions1=jquery@dc1
    # Deliberately changing query labels cannot affect the fit-only solver or predictions.
    changed_query_labels=true_q+rng.normal(scale=10.,size=true_q.shape)
    dc2,diag2=select_fit_ridge_and_refit(jfit,yfit,jfit[train],yfit[train],
        jfit[valid],yfit[valid],ridge_factors=[0.,1e-6,1e-3])
    predictions2=jquery@dc2
    np.testing.assert_array_equal(dc1,dc2)
    np.testing.assert_array_equal(predictions1,predictions2)
    assert diag1["selected_ridge_factor"]==diag2["selected_ridge_factor"]
    assert not np.isclose(np.mean((predictions1-true_q)**2),
                          np.mean((predictions1-changed_query_labels)**2))


def test_known_linear_ground_truth_recovers_one_shared_displacement():
    rng=np.random.default_rng(30)
    a=rng.normal(size=(3,3)); dc_true=np.array([.3,-.2,.1])
    jfit=rng.normal(size=(256,3))@a
    jquery=rng.normal(size=(512,3))@a
    yfit=jfit@dc_true; yquery=jquery@dc_true
    dc,diagnostics=_svd_ridge_solve(jfit,yfit,0.0)
    np.testing.assert_allclose(dc,dc_true,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(jquery@dc,yquery,rtol=1e-10,atol=1e-10)
    assert diagnostics["finite_displacement"]


def test_low_rank_jacobian_uses_finite_svd_ridge_and_reports_rank_deficiency():
    base=np.arange(20,dtype=float)[:,None]
    j=np.concatenate([base,2*base,base],axis=1)
    y=np.sin(base[:,0])
    dc,info=_svd_ridge_solve(j,y,0.0)
    assert np.isfinite(dc).all()
    assert info["rank_deficient"] and info["effective_rank"]==1
    dc_ridge,info_ridge=_svd_ridge_solve(j,y,1e-4)
    assert np.isfinite(dc_ridge).all() and info_ridge["solver"].endswith("no Gram inverse")


def test_nonlinear_counterexample_detects_local_tangent_not_executable_transfer():
    # f(c)=c+10c^2 has unit local derivative at zero but strong curvature.
    c0=0.0; dc=.5; delta=.5
    j_at_c0=1.0
    linear_effect=j_at_c0*dc
    nonlinear_effect=(c0+dc)+10*(c0+dc)**2-(c0+10*c0**2)
    linear_skill=1-(linear_effect-delta)**2/(delta**2)
    nonlinear_skill=1-(nonlinear_effect-delta)**2/(delta**2)
    assert linear_skill==1.0
    assert nonlinear_skill<0.0


def test_physical_context_jacobian_matches_finite_difference_after_denormalization():
    torch.manual_seed(9)
    model=ContextDynamicsModel(context_dim=4,hidden_dim=10,depth=1).eval()
    normalizer=_normalizer(); device=torch.device("cpu")
    c=torch.randn(1,4); probes=np.random.default_rng(8).normal(size=(5,6)).astype(np.float32)
    jac=batched_context_jacobian(model,normalizer,c,probes,device)
    direction=np.array([.2,-.1,.3,.4],dtype=np.float64); eps=2e-3
    prediction=predict_crossfit_query(model,normalizer,c,direction*eps,probes,device)
    prediction_minus=predict_crossfit_query(model,normalizer,c,-direction*eps,probes,device)
    fd=(prediction["pred_new"]-prediction_minus["pred_new"])/(2*eps)
    np.testing.assert_allclose(np.einsum('nod,d->no',jac,direction),fd,rtol=2e-2,atol=2e-5)


def test_fit_query_stage_pair_seeds_and_fingerprints_are_distinct():
    p=sample_environment(make_split_manifest(),"validation",2)
    old,fit,query=stage_pair(p,(3,),seed=7201,n_support=96,n_query=128,dt=.05,sigma=.0005)
    fit_rows=np.column_stack((fit.states[:-1],fit.actions))
    query_rows=np.column_stack((query.states[:-1],query.actions))
    assert not np.array_equal(fit_rows,query_rows)
    assert fit.states[0].tolist()==old.states[-1].tolist()
    assert np.isfinite(fit_rows).all() and np.isfinite(query_rows).all()


def test_opaque_code_map_maps_physical_u4_through_checkpoint_bijection():
    code_map={"0":2,"1":0,"2":3,"3":1}
    assert map_intervention_program((3,),code_map)==(1,)
    assert map_intervention_program((2,),code_map)==(3,)


def test_blocked_inner_split_has_a_gap_and_query_refit_is_marked_leaky():
    train,val=blocked_inner_indices(128,.25,.10)
    assert train[-1]<val[0]-1
    rng=np.random.default_rng(6); j=rng.normal(size=(64,3)); y=rng.normal(size=64)
    _,diag=solve_query_refit_upper_bound(j,y)
    assert "QUERY-LABEL LEAKAGE" in diag["status"]
    assert diag["excluded_from_feasibility_gate"]


def test_prediction_api_has_no_query_label_parameter():
    import inspect
    params=set(inspect.signature(predict_crossfit_query).parameters)
    assert not any("label" in name or "effect" in name or "truth" in name for name in params)


def test_write_once_output_guard_allows_new_or_empty_but_rejects_populated(tmp_path):
    target=tmp_path/"run"
    assert ensure_write_once_directory(target)==target
    assert ensure_write_once_directory(target)==target
    (target/"artifact.json").write_text("{}")
    with pytest.raises(FileExistsError):
        ensure_write_once_directory(target)


def test_missing_required_checkpoints_are_reported_without_substitution(tmp_path):
    present=tmp_path/"present.pt"; present.write_bytes(b"checkpoint")
    absent=tmp_path/"missing.pt"
    assert missing_required_files([present,absent])==[str(absent)]
