import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from rwt.clsm.config_r3 import load_r3_config, validate_r3_config
from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.dynamics_r3 import InterventionResolvedLaw, intervention_channels, validate_r3_checkpoint
from rwt.clsm.schemas import EnvSchema


def toy_schema():
    return EnvSchema("toy", 7, 3, (4, 5, 6), (0, 1, 2), np.eye(3),
                     np.array([0.1, 0.2, 0.3]), np.full(3, -1.), np.full(3, 1.), 0.01)


def toy_partition():
    return torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.bool)


def test_r3_intervention_channels_keep_identity_and_mask_inactive_strengths():
    schema = toy_schema()
    obs = torch.randn(5, 7, dtype=torch.float64)
    action = torch.randn(5, 3, dtype=torch.float64)
    code = torch.tensor([0, 1, 2, 4, 5])
    xi = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    channels = intervention_channels(schema, toy_partition(), obs, action, code, xi)
    assert torch.equal(channels.effective_multipliers[0], torch.ones(3, dtype=torch.float64))
    assert torch.equal(channels.effective_multipliers[1, 1:], torch.ones(2, dtype=torch.float64))
    assert torch.equal(channels.effective_multipliers[2, (0, 2)], torch.ones(2, dtype=torch.float64))
    assert torch.equal(channels.effective_multipliers[3, :2], torch.ones(2, dtype=torch.float64))
    assert torch.equal(channels.eta[4, 1], torch.zeros((), dtype=torch.float64))
    grad = torch.autograd.grad(channels.delta_tau.square().sum(), xi)[0]
    assert torch.equal(grad[0], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(grad[1, 1:], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(grad[2, (0, 2)], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(grad[3, :2], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(grad[4, 1], torch.zeros((), dtype=torch.float64))
    assert torch.all(channels.delta_tau[:, :3] == channels.delta_tau[:, :3])


def test_r3_nominal_anchor_is_exact_and_anchor_is_frozen():
    torch.manual_seed(23)
    schema = toy_schema()
    f0 = ExecutableLaw(schema, hidden_dim=8, partition=toy_partition())
    model = InterventionResolvedLaw(schema, f0, toy_partition(), 12,
                                    torch.zeros(7), torch.ones(7))
    obs = torch.randn(16, 7, dtype=torch.float64)
    action = torch.randn(16, 3, dtype=torch.float64)
    code = torch.zeros(16, dtype=torch.long)
    xi = torch.randn(16, 3, dtype=torch.float64)
    got, _ = model(obs, action, code, xi)
    with torch.no_grad():
        expected, _ = f0(obs, action, code, torch.zeros_like(xi))
        expected = (expected - obs)  # delta_mean=0, delta_scale=1
    assert torch.equal(got, expected)
    assert not any(p.requires_grad for p in model.f0.parameters())
    assert model.trainable_parameter_count() == sum(p.numel() for p in model.residual.parameters()) + 7


def test_r3_rejects_legacy_checkpoint_payloads():
    with pytest.raises(ValueError, match="rejects R2"):
        validate_r3_checkpoint({"architecture_revision": 2, "architecture_id": "clsm_r2"})
    validate_r3_checkpoint({"architecture_revision": 3,
                            "architecture_id": InterventionResolvedLaw.architecture_id,
                            "role": "oracle_diagnostic", "diagnostic_id": "CLSM-R3-R1"})


def test_r3_config_is_frozen_and_legacy_revision_rejected():
    config_path = Path(__file__).resolve().parents[2] / "experiments/rwt/clsm/configs/clsm_r3_final.yaml"
    config = load_r3_config(config_path)
    legacy = copy.deepcopy(config)
    legacy["architecture_revision"] = 2
    with pytest.raises(ValueError, match="rejects legacy"):
        validate_r3_config(legacy)


def test_equal_total_force_can_have_distinct_channel_decompositions_at_zero_velocity():
    schema = toy_schema()
    partition = toy_partition()
    obs = torch.zeros((1, 7), dtype=torch.float64)  # joint velocities are zero
    # Motor-only and damping-only channels can have equal total force while
    # retaining distinct eta/delta_tau mechanism identities.
    motor_action = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    damping_action = torch.tensor([[0.675, 0., 0.]], dtype=torch.float64)
    motor_code = torch.tensor([1]); damping_code = torch.tensor([4])
    motor_xi = torch.tensor([[0., 0., 0.]], dtype=torch.float64)
    damp_xi = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    motor = intervention_channels(schema, partition, obs, motor_action, motor_code, motor_xi)
    damp = intervention_channels(schema, partition, obs, damping_action, damping_code, damp_xi)
    motor_total = motor.nominal_force + motor.delta_tau.reshape(1, 3, 3).sum(1)
    damp_total = damp.nominal_force + damp.delta_tau.reshape(1, 3, 3).sum(1)
    assert torch.allclose(motor_total, damp_total)
    assert not torch.equal(torch.cat((motor.eta, motor.delta_tau), 1),
                           torch.cat((damp.eta, damp.delta_tau), 1))
