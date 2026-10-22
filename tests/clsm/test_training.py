import numpy as np
import torch
import gymnasium
import mujoco
import platform
import scipy

from rwt.clsm.belief import LawFilter
from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.evolution import LawEvolution
from rwt.clsm.factory import build_clsm
from rwt.clsm.memory import HierarchyParameters, TheoryMemory
from rwt.clsm.schemas import EnvSchema
from rwt.clsm.config import load_config
from rwt.clsm.train import (EmissionStatistics, PublicSequence, approximate_estep,
                            collect_emission_statistics, expected_emission_nll,
                            expected_evolution_nll, generalized_em_mstep,
                            make_mstep_optimizer)
from rwt.clsm.schemas import mujoco_nominal_schema
from experiments.rwt.clsm.run_train import _load_checkpoint, _save_checkpoint


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
    assert len(result.inference_diagnostics) == len(obs)
    assert len(result.evolution_diagnostics) == len(obs)
    assert abs(sum(result.inference_diagnostics[0][f"code_weight_{code}"]
                   for code in range(8)) - 1) < 1e-12
    evolution_row = result.evolution_diagnostics[0]
    assert abs(evolution_row["expected_switch_probability"] +
               evolution_row["expected_parameter_reset_probability"] +
               evolution_row["expected_drift_probability"] - 1) < 1e-12
    assert abs(sum(evolution_row[f"switch_mass_hamming_{distance}"]
                   for distance in (1, 2, 3)) -
               evolution_row["expected_switch_probability"]) < 1e-12


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


def test_emission_weighted_batch_gradient():
    class LinearNll(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slope = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))

        def log_prob(self, obs, action, next_obs, code, theta):
            return -self.slope * obs[:, 0]

    stats = EmissionStatistics(
        obs=torch.tensor([[0.], [10.]], dtype=torch.float64),
        action=torch.zeros((2, 1), dtype=torch.float64),
        next_obs=torch.zeros((2, 1), dtype=torch.float64),
        code=torch.zeros(2, dtype=torch.long),
        theta=torch.zeros((2, 3), dtype=torch.float64),
        cov=torch.zeros((2, 3, 3), dtype=torch.float64),
        weight=torch.tensor([.99, .01], dtype=torch.float64),
    )
    model = LinearNll()

    def gradient(indices=None, **kwargs):
        loss = expected_emission_nll(model, stats, indices, **kwargs)
        return float(torch.autograd.grad(loss, model.slope)[0])

    exact = gradient()
    assert abs(exact - .1) < 1e-12
    uniform_expectation = sum(gradient(torch.tensor([i])) for i in range(2)) / 2
    assert abs(uniform_expectation - exact) < 1e-12
    probabilities = stats.weight / stats.weight.sum()
    weighted_expectation = sum(float(probabilities[i]) * gradient(
        torch.tensor([i]), sampled_by_weight=True) for i in range(2))
    assert abs(weighted_expectation - exact) < 1e-12


def test_checkpoint_roundtrip_restores_optimizer_and_rng_for_resume(tmp_path):
    config = load_config("experiments/rwt/clsm/configs/cpu_smoke.yaml")
    parts = build_clsm(mujoco_nominal_schema("Reacher-v5"), config)
    optimizer = make_mstep_optimizer(parts.law, parts.evolution, parts.hierarchy, config)
    generator = torch.Generator().manual_seed(407)
    optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in parts.law.parameters()).backward()
    optimizer.step()
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__}
    path = tmp_path / "resume.pt"
    hashes = {"train": "train-hash", "validation": "val-hash"}
    _save_checkpoint(path, parts, optimizer, generator, config, hashes,
                     outer=1, best_nll=2.5, versions=versions,
                     commit="test-commit", source_dirty=False)

    torch.rand(4, generator=generator)
    multiplier_expected = torch.rand((), generator=generator)
    optimizer.zero_grad()
    (parts.law.log_variance.square().sum() * multiplier_expected).backward()
    optimizer.step()
    expected = {name: value.clone() for name, value in parts.law.state_dict().items()}

    restored = build_clsm(mujoco_nominal_schema("Reacher-v5"), config)
    restored_optimizer = make_mstep_optimizer(restored.law, restored.evolution,
                                               restored.hierarchy, config)
    restored_generator = torch.Generator().manual_seed(0)
    _load_checkpoint(path, restored, restored_optimizer, restored_generator, config, hashes)
    torch.rand(4, generator=restored_generator)
    multiplier_actual = torch.rand((), generator=restored_generator)
    restored_optimizer.zero_grad()
    (restored.law.log_variance.square().sum() * multiplier_actual).backward()
    restored_optimizer.step()
    assert torch.equal(multiplier_expected, multiplier_actual)
    for name, value in restored.law.state_dict().items():
        assert torch.equal(value, expected[name])
