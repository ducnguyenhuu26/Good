import numpy as np
import pytest
import torch

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


def test_resampling_mass_and_support():
    zero = torch.zeros(3,dtype=torch.float64)
    cov = torch.zeros((3,3),dtype=torch.float64)
    items = [LawComponent(0,i,zero,cov,torch.tensor(float(x),dtype=torch.float64))
             for i,x in enumerate((0.,-1.,-2.))]
    kept,dropped = compress_components(items,2)
    assert 0 < dropped < 1
    assert len(kept)==2
    assert abs(sum(float(x.log_weight.exp()) for x in kept)-1)<1e-12
