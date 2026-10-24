import numpy as np
import pytest
import torch

from q1.model import TransitionNormalizer
from q1.simulator import DynamicsParams, apply_program, response
from q1.train_primitives import (TransferTask, _balanced_after_effect_loss,
                                 _paired_counterfactual_targets, _paired_effect_scales)


def _task(group=2):
    before = DynamicsParams(np.array([[0.9, 0.08], [-0.05, 0.86]]), 0.34, 0.11, 0.23)
    after = apply_program(before, (group,))
    x = np.array([[0.2, -0.1, 0.3, -0.2, 0.8, 0.1],
                  [-0.3, 0.25, -0.15, 0.22, -0.4, 0.7]], dtype=np.float32)
    y = response(after, x, dt=0.05).astype(np.float32)
    query = np.concatenate((x, y), axis=1)
    return TransferTask(query.copy(), query, before, after, group, (group,), "train-0000")


def test_paired_targets_are_same_probe_simulator_counterfactuals():
    task = _task(2)
    after, effect = _paired_counterfactual_targets(task, task.query[:, :6], dt=0.05,
                                                   device=torch.device("cpu"))
    expected_after = response(task.target_params, task.query[:, :6], dt=0.05)
    expected_before = response(task.before_params, task.query[:, :6], dt=0.05)
    assert np.allclose(after.numpy(), expected_after)
    assert np.allclose(effect.numpy(), expected_after - expected_before)


def test_effect_scales_are_train_only_and_floor_small_code_energy():
    prepared = [(_task(code), torch.zeros(1, 2), torch.zeros(2, 6), torch.zeros(2, 2))
                for code in range(4)]
    scales = _paired_effect_scales(prepared, dt=0.05, scale_floor_fraction=0.01)
    assert scales["global_after_energy"] > 0
    assert set(scales["per_code_effect_energy"]) == {"0", "1", "2", "3"}
    assert all(value >= scales["scale_floor"]
               for value in scales["per_code_effect_scale_used"].values())
    assert scales["per_code_effect_scale_used"]["3"] >= scales["per_code_effect_energy"]["3"]


def test_balanced_after_effect_loss_beta_zero_is_after_only_and_effect_is_group_balanced():
    pred_after = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    pred_before = torch.zeros_like(pred_after)
    target_after = torch.zeros_like(pred_after)
    target_effect = torch.ones_like(pred_after)
    group_ids = torch.tensor([0, 0, 1, 1])
    scales = {str(i): 1.0 for i in range(4)}
    with pytest.raises(ValueError, match="missing training intervention group"):
        _balanced_after_effect_loss(pred_after, pred_before, target_after, target_effect,
                                    group_ids, 1.0, scales, beta=0.0)
    balanced_groups = torch.tensor([0, 1, 2, 3])
    loss0, after0, effect0 = _balanced_after_effect_loss(
        pred_after, pred_before, target_after, target_effect, balanced_groups, 1.0,
        scales, beta=0.0)
    loss1, after1, effect1 = _balanced_after_effect_loss(
        pred_after, pred_before, target_after, target_effect, balanced_groups, 1.0,
        scales, beta=0.25)
    assert torch.allclose(loss0, after0)
    assert torch.allclose(after0, after1)
    assert torch.allclose(loss1, after1 + 0.25 * effect1)
    assert effect0 > 0 and effect1 > 0
