import numpy as np
import pytest
import torch

from experiments.rwt.run_stage3a_final_cpu import context_for
from rwt.stage3a import CEMConfig, MiniCEM, MiniDynamics
from rwt.stage3a_diagnostic import (HISTORY_STEPS, HISTORY_WIDTH,
                                     ConditionedMiniDynamics, TrainSupport, append_history,
                                     observed_histories)


def test_history_uses_only_prior_transitions_and_resets_at_episode_boundary():
    state = np.zeros((4, 17), np.float32)
    action = np.zeros((4, 6), np.float32)
    action[:, 0] = [1, 2, 3, 4]
    next_state = np.ones((4, 17), np.float32)
    data = {"state": state, "action": action, "next_state": next_state,
            "terminated": np.array([False, True, False, False]),
            "truncated": np.zeros(4, bool)}
    history = observed_histories(data)
    assert history.shape == (4, HISTORY_STEPS * HISTORY_WIDTH)
    assert np.all(history[0] == 0) and np.all(history[2] == 0)
    assert history[1, 0] == 1 and history[3, 0] == 3
    data["next_state"][3] = 1000
    np.testing.assert_array_equal(observed_histories(data), history)
    with pytest.raises(ValueError):
        append_history(np.zeros(4), state[0], action[0], next_state[0])


def test_conditioned_model_shapes_and_gradients():
    model = ConditionedMiniDynamics(3)
    state, action = torch.randn(5, 17), torch.randn(5, 6)
    context = torch.eye(3)[[0, 1, 2, 0, 1]]
    next_state, reward = model(state, action, context)
    assert next_state.shape == (5, 17) and reward.shape == (5,)
    model.loss(state, action, state, torch.zeros(5), context).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    with pytest.raises(ValueError):
        model(state, action, torch.zeros(5, 4))


def test_oracle_context_is_evaluator_side_only_and_uses_abd():
    template = {"state": np.zeros((2, 17)), "action": np.zeros((2, 6)),
                "next_state": np.zeros((2, 17)), "terminated": np.zeros(2, bool),
                "truncated": np.array([False, True])}
    data = {r: template for r in ("A", "B", "D")}
    assert context_for(data, "pooled") is None
    np.testing.assert_array_equal(context_for(data, "oracle"), np.repeat(np.eye(3), 2, axis=0))
    assert context_for(data, "history").shape == (6, HISTORY_STEPS * HISTORY_WIDTH)


def test_cem_diagnostic_plan_matches_its_own_reward_sum():
    model = MiniDynamics()
    planner = MiniCEM(model, -np.ones(6), np.ones(6), CEMConfig(3, 8, 2),
                      seed=4, diagnostics=True)
    action = planner.act(np.zeros(17))
    assert planner.last_plan_actions.shape == (3, 6)
    assert planner.last_plan_states.shape == (3, 17)
    assert planner.last_plan_rewards.shape == (3,)
    np.testing.assert_array_equal(action, planner.last_plan_actions[0])
    np.testing.assert_allclose(planner.last_predicted_return,
                               planner.last_plan_rewards.sum(), atol=1e-6)


def test_support_reference_uses_train_and_validation_only():
    rng = np.random.default_rng(7)
    state = rng.normal(size=(40, 17)).astype(np.float32)
    action = rng.normal(size=(40, 6)).astype(np.float32)
    support = TrainSupport(state, action, state[:10], action[:10])
    assert support.threshold >= 0
    assert support.distance(state[:1], action[:1])[0] < 1e-2
    far = support.distance(np.full((1, 17), 100, np.float32), np.zeros((1, 6), np.float32))[0]
    assert far > support.threshold
