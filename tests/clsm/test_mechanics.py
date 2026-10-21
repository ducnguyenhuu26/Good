import numpy as np
import pytest
import torch

from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.schemas import EnvSchema, mujoco_nominal_schema


def schema():
    return EnvSchema("toy", 3, 2, (1, 2), (0, 1), np.eye(2),
                     np.ones(2), -np.ones(2), np.ones(2), 0.01)


def test_nominal_anchor_and_inactive_theta():
    law = ExecutableLaw(schema(), hidden_dim=8)
    obs = torch.tensor([[0., 1., -2.]], dtype=torch.float64)
    action = torch.ones((1, 2), dtype=torch.float64)
    zero = torch.zeros((1, 3), dtype=torch.float64)
    changed = torch.full((1, 3), 100., dtype=torch.float64)
    nominal = law.mechanics(obs, action, torch.tensor([0]), zero)
    other = law.mechanics(obs, action, torch.tensor([0]), changed)
    assert torch.equal(nominal[1], action)
    assert torch.equal(nominal[1], other[1])
    assert torch.equal(nominal[2], other[2])
    assert torch.allclose(law.mechanics.W.norm(dim=0), torch.ones(2, dtype=torch.float64))
    assert not torch.equal(law.mechanics.W[:, 0], law.mechanics.W[:, 1])


def test_mask_paths_and_dissipation():
    law = ExecutableLaw(schema(), hidden_dim=8)
    obs = torch.tensor([[0., 1., -2.]], dtype=torch.float64)
    action = torch.ones((1, 2), dtype=torch.float64)
    theta = torch.zeros((1, 3), dtype=torch.float64)
    nominal = law.mechanics(obs, action, torch.tensor([0]), theta)
    motor = law.mechanics(obs, action, torch.tensor([1]), theta)
    damping = law.mechanics(obs, action, torch.tensor([4]), theta)
    assert torch.equal(motor[2], nominal[2])
    assert torch.equal(damping[1], nominal[1])
    assert bool(((obs[:, 1:] * damping[2]).sum(-1) <= 0).all())
    assert bool((motor[1] < nominal[1]).all())


def test_actuator_gradient_and_no_action_bypass():
    law = ExecutableLaw(schema(), hidden_dim=8)
    obs = torch.tensor([[0., 0., 0.]], dtype=torch.float64)
    action = torch.ones((1, 2), dtype=torch.float64)
    theta = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    mean = law(obs, action, torch.tensor([1]), theta)[0]
    grad = torch.autograd.grad(mean.sum(), theta)[0]
    assert torch.isfinite(grad).all() and grad[0, 0].abs() > 0
    assert torch.equal(grad[0, 1:], torch.zeros(2, dtype=torch.float64))
    # Both MLP inputs are state and effective effort, never raw action.
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
