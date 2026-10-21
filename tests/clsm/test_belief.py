import numpy as np
import pytest
import torch
from types import SimpleNamespace

from rwt.clsm.belief import LawComponent, LawFilter, compress_components, ekf_correct
from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.evolution import LawEvolution
from rwt.clsm.memory import TheoryMemory
from rwt.clsm.schemas import EnvSchema


def fixture():
    schema = EnvSchema("toy", 3, 2, (1,2), (0,1), np.eye(2),
                       np.ones(2), -np.ones(2), np.ones(2), .01)
    return ExecutableLaw(schema, hidden_dim=8), LawEvolution(), TheoryMemory()


def test_parameter_reset_age():
    _, evolution, memory = fixture()
    mean, cov = memory.prior_for_entry(1)
    item = LawComponent(1, 4, mean, cov, torch.tensor(0., dtype=torch.float64))
    branches = evolution.predict_branches(item, memory)
    assert next(b for b in branches if b.event == "stay_reset").age == 5
    assert all(b.age == 0 for b in branches if b.event == "switch")
    assert abs(sum(b.probability for b in branches) - 1) < 1e-12


def test_filter_observe_then_evolve():
    model, evolution, memory = fixture()
    filt = LawFilter(model, evolution, memory)
    obs = torch.zeros(3, dtype=torch.float64)
    action = torch.ones(2, dtype=torch.float64)
    predicted = model(obs[None], action[None], torch.tensor([0]), torch.zeros((1,3),dtype=torch.float64))[0][0].detach()
    state = filt.observe(obs, action, predicted)
    assert state["time"] == 0 and filt.time == 0
    assert torch.allclose(filt.memory.log_model_weights.exp().sum(),
                          torch.tensor(1., dtype=torch.float64))
    assert all(c.age == 0 for c in filt.components)
    ancestry = filt.evolve()
    assert filt.time == 1 and all(c.age in (0,1) for c in filt.components)
    assert abs(sum(float(c.log_weight.exp()) for c in filt.components) - 1) < 1e-10
    assert torch.allclose(ancestry.filtered_weight @ ancestry.transition,
                          ancestry.predicted_weight, atol=1e-10)
    assert torch.allclose(sum(ancestry.event_transition.values()), ancestry.transition, atol=1e-12)
    assert 0 <= filt.diagnostics[-1]["completed_visit_mass"] <= 1
    visits = filt.memory.completed_visits
    with pytest.raises(RuntimeError, match="observe"):
        filt.evolve()
    assert filt.memory.completed_visits == visits


def test_joseph_covariance_and_nominal_empty_subspace():
    model, _, memory = fixture()
    for code in (0,7):
        mean,cov = memory.prior_for_entry(code)
        item = LawComponent(code,0,mean,cov,torch.tensor(0.,dtype=torch.float64))
        updated, likelihood, _ = ekf_correct(model,item,torch.zeros(3,dtype=torch.float64),
                                               torch.ones(2,dtype=torch.float64),torch.zeros(3,dtype=torch.float64))
        assert torch.isfinite(likelihood)
        assert torch.allclose(updated.cov,updated.cov.T,atol=1e-12)
        assert torch.linalg.eigvalsh(updated.cov).min() >= -1e-12
        if code == 0:
            assert torch.equal(updated.mean,torch.zeros(3,dtype=torch.float64))


def test_ekf_linear_gaussian_matches_closed_form():
    class LinearLaw:
        schema = SimpleNamespace(observation_dim=1, action_dim=1)

        def __call__(self, obs, action, code, theta):
            mean = obs + 2.0 * theta[:, :1]
            variance = torch.full_like(mean, .2)
            return mean, variance

    prior_mean, prior_var = .3, .4
    component = LawComponent(1, 0, torch.tensor([prior_mean, 0., 0.], dtype=torch.float64),
                             torch.diag(torch.tensor([prior_var, 0., 0.], dtype=torch.float64)),
                             torch.tensor(0., dtype=torch.float64))
    obs = torch.tensor([.3], dtype=torch.float64)
    next_obs = torch.tensor([1.4], dtype=torch.float64)
    corrected, log_evidence, _ = ekf_correct(LinearLaw(), component, obs,
        torch.zeros(1, dtype=torch.float64), next_obs)

    residual = float(next_obs[0] - (obs[0] + 2 * prior_mean))
    innovation_var = 4 * prior_var + .2
    gain = prior_var * 2 / innovation_var
    expected_mean = prior_mean + gain * residual
    expected_var = (1 - gain * 2) ** 2 * prior_var + gain**2 * .2
    expected_log_evidence = -.5 * (residual**2 / innovation_var + np.log(innovation_var) + np.log(2*np.pi))
    assert abs(float(corrected.mean[0]) - expected_mean) < 1e-12
    assert abs(float(corrected.cov[0, 0]) - expected_var) < 1e-12
    assert abs(float(log_evidence) - expected_log_evidence) < 1e-12


def test_filter_state_roundtrip_preserves_next_update():
    model, evolution, memory = fixture()
    filt = LawFilter(model, evolution, memory)
    obs = torch.zeros(3, dtype=torch.float64)
    action = torch.ones(2, dtype=torch.float64)
    for _ in range(2):
        filt.observe(obs, action, obs + .01)
        filt.evolve()
    snapshot = filt.state_dict()

    restored_model, restored_evolution, restored_memory = fixture()
    restored_model.load_state_dict(model.state_dict())
    restored_evolution.load_state_dict(evolution.state_dict())
    restored = LawFilter(restored_model, restored_evolution, restored_memory)
    restored.load_state_dict(snapshot)
    assert restored.time == filt.time
    assert len(restored.components) == len(filt.components)
    for left, right in zip(restored.components, filt.components):
        assert left.code == right.code and left.age == right.age
        assert left.scratch_or_reuse == right.scratch_or_reuse
        assert torch.equal(left.mean, right.mean)
        assert torch.equal(left.cov, right.cov)
        assert torch.equal(left.log_weight, right.log_weight)

    left = filt.observe(obs, action, obs + .02)
    right = restored.observe(obs, action, obs + .02)
    assert abs(left["log_predictive"] - right["log_predictive"]) < 1e-12


def test_resampling_mass_and_support():
    zero = torch.zeros(3,dtype=torch.float64)
    cov = torch.zeros((3,3),dtype=torch.float64)
    items = [LawComponent(0,i,zero,cov,torch.tensor(float(x),dtype=torch.float64))
             for i,x in enumerate((0.,-1.,-2.))]
    kept,dropped = compress_components(items,2)
    assert 0 < dropped < 1
    assert len(kept)==2
    assert abs(sum(float(x.log_weight.exp()) for x in kept)-1)<1e-12


def test_scratch_only_filter_keeps_zero_reuse_mass_without_nan():
    model, evolution, _ = fixture()
    filt = LawFilter(model, evolution, TheoryMemory(scratch_weight=1.))
    obs = torch.zeros(3, dtype=torch.float64)
    action = torch.ones(2, dtype=torch.float64)
    for _ in range(2):
        result = filt.observe(obs, action, obs)
        assert np.isfinite(result["log_predictive"])
        assert torch.equal(filt.memory.log_model_weights.exp(),
                           torch.tensor([1., 0.], dtype=torch.float64))
        filt.evolve()
    assert filt.memory.scratch_theory_mean  # The scratch model adapted from these observed transitions.


def test_scratch_hypothesis_matches_standalone_scratch_on_same_history():
    mixed_model, mixed_evolution, _ = fixture()
    scratch_model, scratch_evolution, _ = fixture()
    scratch_model.load_state_dict(mixed_model.state_dict())
    scratch_evolution.load_state_dict(mixed_evolution.state_dict())
    mixed = LawFilter(mixed_model, mixed_evolution, TheoryMemory(scratch_weight=.5))
    scratch = LawFilter(scratch_model, scratch_evolution, TheoryMemory(scratch_weight=1.))
    obs = torch.zeros(3, dtype=torch.float64)
    action = torch.tensor([.5, -.2], dtype=torch.float64)
    for scale in (.01, -.02, .015):
        next_obs = obs + scale
        mixed.observe(obs, action, next_obs); mixed.evolve()
        scratch.observe(obs, action, next_obs); scratch.evolve()
        assert torch.equal(mixed.memory.scratch_primitive_mean, scratch.memory.scratch_primitive_mean)
        assert torch.equal(mixed.memory.scratch_primitive_var, scratch.memory.scratch_primitive_var)
        assert mixed.memory.scratch_theory_mean.keys() == scratch.memory.scratch_theory_mean.keys()
        for code in mixed.memory.scratch_theory_mean:
            assert torch.allclose(mixed.memory.scratch_theory_mean[code],
                                  scratch.memory.scratch_theory_mean[code], atol=1e-12)
            assert torch.allclose(mixed.memory.scratch_theory_var[code],
                                  scratch.memory.scratch_theory_var[code], atol=1e-12)


def test_zero_mass_branch_never_occupies_support_slot():
    zero = torch.zeros(3, dtype=torch.float64)
    cov = torch.zeros((3, 3), dtype=torch.float64)
    components = [LawComponent(0, 0, zero, cov, torch.tensor(0., dtype=torch.float64)),
                  LawComponent(7, 0, zero, cov, torch.tensor(-torch.inf, dtype=torch.float64))]
    kept, dropped = compress_components(components, 1)
    assert len(kept) == 1 and kept[0].code == 0 and dropped == 0
