import numpy as np
import pytest
import torch
from pathlib import Path
from scipy.stats import kstest

from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.schemas import EnvSchema, mujoco_nominal_schema
from rwt.clsm.support import DisjointPartition, bounded_physical_strength


def schema():
    return EnvSchema("toy", 3, 2, (1, 2), (0, 1), np.eye(2),
                     np.ones(2), -np.ones(2), np.ones(2), 0.01)


def test_disjoint_partition_enumeration_and_canonical_permutation():
    candidates = DisjointPartition.enumerate(6)
    assert len(candidates) == 31
    keys = set()
    for mask in candidates:
        assert mask.dtype == torch.bool
        assert torch.equal(mask.sum(1), torch.ones(6, dtype=torch.long))
        assert bool((mask.sum(0) > 0).all())
        assert bool(mask[0, 0])
        keys.add(tuple(mask.flatten().tolist()))
    assert len(keys) == len(candidates)
    swapped = candidates[7].flip(1)
    assert torch.equal(DisjointPartition.canonical(swapped), candidates[7])
    source = (Path(__file__).resolve().parents[2] / "src/rwt/clsm/support.py").read_text(encoding="utf-8")
    assert "physics_wrappers" not in source and "clsm_evaluator" not in source


def test_standard_normal_strength_prior_maps_to_registered_uniform_interval():
    rng = np.random.default_rng(581)
    xi = torch.tensor(rng.normal(size=(20000, 3)), dtype=torch.float64)
    motor, damping = bounded_physical_strength(xi)
    motor_unit = ((motor[:, 0].numpy() - .50) / .35)
    damping_unit = ((damping[:, 0].numpy() - 1.30) / .70)
    assert kstest(motor_unit, "uniform").statistic < .015
    assert kstest(damping_unit, "uniform").statistic < .015


def test_partition_and_motor_label_permutation_preserve_physical_force():
    left = torch.tensor([[True, False], [True, False], [False, True]])
    right = left.flip(1)
    law_left = ExecutableLaw(EnvSchema("perm", 4, 3, (1, 2, 3), (0, 1, 2),
                                      np.eye(3), np.ones(3), -np.ones(3), np.ones(3), .01),
                             hidden_dim=8, partition=left)
    law_right = ExecutableLaw(law_left.schema, hidden_dim=8, partition=right)
    obs = torch.tensor([[0., .3, -.4, .8]], dtype=torch.float64)
    action = torch.tensor([[.2, -.1, .7]], dtype=torch.float64)
    xi = torch.tensor([[.4, -.3, .2]], dtype=torch.float64)
    force_left = law_left.mechanics(obs, action, torch.tensor([1]), xi)[0]
    # Constructors canonicalize the swapped global representation once, so
    # public code bits and strength coordinates retain their canonical labels.
    force_right = law_right.mechanics(obs, action, torch.tensor([1]), xi)[0]
    assert torch.allclose(force_left, force_right, atol=1e-12, rtol=0)


def test_bounded_physical_strength_transform_and_inactive_coordinates():
    values = torch.tensor([[-40., 0., 40.], [40., 0., -40.]], dtype=torch.float64)
    motor, damping = bounded_physical_strength(values)
    assert bool(((motor >= .50) & (motor <= .85)).all())
    assert bool(((damping >= 1.30) & (damping <= 2.00)).all())
    assert torch.allclose(motor[0], torch.tensor([.50, .675], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(damping[0], torch.tensor([2.0], dtype=torch.float64), atol=1e-12)


def test_nominal_force_and_bounded_mechanism_paths():
    partition = torch.tensor([[True, False], [False, True]])
    law = ExecutableLaw(schema(), hidden_dim=8, partition=partition)
    obs = torch.tensor([[0., 1., -2.]], dtype=torch.float64)
    action = torch.ones((1, 2), dtype=torch.float64)
    xi = torch.zeros((1, 3), dtype=torch.float64)
    nominal = law.mechanics(obs, action, torch.tensor([0]), xi)
    changed_inactive = law.mechanics(obs, action, torch.tensor([0]),
                                     torch.full((1, 3), 100., dtype=torch.float64))
    assert torch.equal(nominal[1], action)
    assert torch.equal(nominal[1], changed_inactive[1])
    assert torch.equal(nominal[2], changed_inactive[2])
    assert torch.equal(law.mechanics.W[:, 0] @ law.mechanics.W[:, 1], torch.tensor(0., dtype=torch.float64))

    motor = law.mechanics(obs, action, torch.tensor([1]), xi)
    damping = law.mechanics(obs, action, torch.tensor([4]), xi)
    assert torch.allclose(motor[1][0, 0], torch.tensor(.675, dtype=torch.float64))
    assert torch.equal(damping[1], nominal[1])
    assert torch.allclose(damping[2], 1.65 * nominal[2])


def test_inactive_strength_has_zero_effect_and_gradient():
    partition = torch.tensor([[True, False], [False, True]])
    law = ExecutableLaw(schema(), hidden_dim=8, partition=partition)
    obs = torch.tensor([[0., 1., -2.]], dtype=torch.float64)
    action = torch.tensor([[.7, -.4]], dtype=torch.float64)
    xi = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    mean = law(obs, action, torch.tensor([1]), xi)[0]
    grad = torch.autograd.grad(mean.sum(), xi)[0]
    assert torch.isfinite(grad).all()
    assert grad[0, 0].abs() > 0
    assert torch.equal(grad[0, 1:], torch.zeros(2, dtype=torch.float64))
    assert law.net[0].in_features == schema().observation_dim + 2


@pytest.mark.parametrize("env_id,dim,act", [("HalfCheetah-v5",17,6), ("Walker2d-v5",17,6), ("Reacher-v5",10,2)])
def test_forward_shapes_all_valid_envs(env_id, dim, act):
    s = mujoco_nominal_schema(env_id)
    law = ExecutableLaw(s, hidden_dim=8)
    obs = torch.zeros((2, dim), dtype=torch.float64)
    action = torch.zeros((2, act), dtype=torch.float64)
    mean, var = law(obs, action, torch.tensor([0,7]), torch.zeros((2,3), dtype=torch.float64))
    assert mean.shape == var.shape == (2,dim)
    assert torch.isfinite(mean).all() and (var > 0).all()


def test_swimmer_damping_exclusion():
    with pytest.raises(ValueError, match="damping"):
        mujoco_nominal_schema("Swimmer-v5")
