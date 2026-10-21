import numpy as np
import torch

from rwt.clsm.belief import LawFilter
from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.evolution import LawEvolution
from rwt.clsm.memory import HierarchyParameters, TheoryMemory
from rwt.clsm.schemas import EnvSchema
from rwt.clsm.train import (PublicSequence, approximate_estep,
                            collect_emission_statistics, expected_emission_nll,
                            expected_evolution_nll, generalized_em_mstep)


def test_public_sequence_estep_event_masses_and_timing():
    schema = EnvSchema("toy", 3, 2, (1, 2), (0, 1), np.eye(2),
                       np.ones(2), -np.ones(2), np.ones(2), .01)
    model = ExecutableLaw(schema, hidden_dim=8)
    filt = LawFilter(model, LawEvolution(), TheoryMemory(), budget=128)
    obs = torch.zeros((2, 3), dtype=torch.float64)
    action = torch.ones((2, 2), dtype=torch.float64)
    next_obs = torch.zeros((2, 3), dtype=torch.float64)
    result = approximate_estep(filt, obs, action, next_obs)
    assert result.log_predictive.shape == (2,)
    assert torch.isfinite(result.log_predictive).all()
    assert len(result.ancestry) == 1
    assert len(result.gamma) == 2
    assert all(abs(float(x.sum()) - 1) < 1e-10 for x in result.gamma)
    assert abs(sum(float(x.sum()) for x in result.smoothed.event_xi[0].values()) - 1) < 1e-10
    assert filt.time == 1 and filt.needs_evolution
    assert result.max_dropped_mass < 1e-3


def test_mstep_routes_finite_gradients_through_emission_and_evolution():
    schema = EnvSchema("toy", 3, 2, (1, 2), (0, 1), np.eye(2),
                       np.ones(2), -np.ones(2), np.ones(2), .01)
    model = ExecutableLaw(schema, hidden_dim=8)
    evolution = LawEvolution()
    obs = torch.tensor([[.1, .2, -.1], [.2, .1, -.2]], dtype=torch.float64)
    action = torch.tensor([[.5, -.3], [.4, .2]], dtype=torch.float64)
    next_obs = obs + .01
    sequence = PublicSequence(obs, action, next_obs)
    posterior = approximate_estep(LawFilter(model, evolution, TheoryMemory()), obs, action, next_obs)
    stats = collect_emission_statistics([sequence], [posterior])
    emission = expected_emission_nll(model, stats)
    transition = expected_evolution_nll(evolution, [posterior])
    (emission + transition).backward()
    assert torch.isfinite(model.net[0].weight.grad).all()
    assert torch.isfinite(model.mechanics.W_raw.grad).all()
    assert torch.isfinite(evolution.log_kappa0.grad)
    assert torch.isfinite(evolution.q_raw.grad).all()
    config = {"training": {"lr_dynamics": .0003, "lr_evolution": .001,
                           "weight_decay": .0001, "batch_size": 8,
                           "gradient_clip_norm": 5., "gradient_steps_per_iteration": 1,
                           "variance_penalty_weight": .01},
              "mechanisms": {"sparse_weight": .001, "overlap_weight": .01}}
    hierarchy = HierarchyParameters()
    metrics = generalized_em_mstep(model, evolution, hierarchy, stats, [posterior], config, seed=41)
    assert len(metrics) == 1 and all(np.isfinite(x) for x in metrics[0].values())
    assert torch.isfinite(hierarchy.recurrence_raw.grad)
