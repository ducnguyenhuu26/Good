import inspect

import numpy as np
import pytest
import torch

from rwt.stage3a import (CEMConfig, DEFAULT_REGIMES, LinearResidualAdapter, MiniCEM,
                         MiniDynamics, RecurrentHalfCheetah,
                         load_checkpoint, save_checkpoint)
from experiments.rwt.run_stage3a import _collect_one, _load_pooled


def test_halfcheetah_dimensions_switching_and_physics_restoration():
    sequence = [("A", 2), ("B", 1), ("D", 1), ("C", 1), ("A", 1), ("C", 1)]
    env = RecurrentHalfCheetah(sequence)
    model = env.unwrapped.model
    original_friction = env._base_friction.copy()
    original_gain = env._base_gain.copy()
    try:
        observation, info = env.reset(seed=123)
        assert observation.shape == (17,) and env.action_space.shape == (6,)
        assert "regime" not in info
        visited = []
        for _ in range(sum(length for _, length in sequence)):
            observation, _, terminated, truncated, info = env.step(np.zeros(6))
            visited.append(env.last_transition_regime)
            assert "regime" not in info and observation.shape == (17,)
            assert not terminated
        assert visited == ["A", "A", "B", "D", "C", "A", "C"]
        assert truncated
        with pytest.raises(RuntimeError):
            env.step(np.zeros(6))
        env.reset(seed=123)
        np.testing.assert_array_equal(model.geom_friction, original_friction)
        np.testing.assert_array_equal(model.actuator_gainprm, original_gain)
        env._apply_regime("B")
        np.testing.assert_allclose(model.geom_friction, original_friction * 0.6)
        np.testing.assert_array_equal(model.actuator_gainprm, original_gain)
        env._apply_regime("D")
        np.testing.assert_array_equal(model.geom_friction, original_friction)
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], original_gain[:, 0] * 0.7)
    finally:
        env.close()
    np.testing.assert_array_equal(model.geom_friction, original_friction)
    np.testing.assert_array_equal(model.actuator_gainprm, original_gain)


def test_halfcheetah_seed_replay_and_perturbations_change_trajectory():
    actions = np.random.default_rng(5).uniform(-0.8, 0.8, size=(24, 6))
    trajectories = []
    for regime in ("A", "A", "B", "D"):
        env = RecurrentHalfCheetah([(regime, len(actions))])
        try:
            state, _ = env.reset(seed=71)
            trace = [state.copy()]
            for action in actions:
                state, _, _, _, _ = env.step(action)
                trace.append(state.copy())
            trajectories.append(np.asarray(trace))
        finally:
            env.close()
    np.testing.assert_array_equal(trajectories[0], trajectories[1])
    assert np.max(np.abs(trajectories[0] - trajectories[2])) > 1e-5
    assert np.max(np.abs(trajectories[0] - trajectories[3])) > 1e-5


def test_model_gradient_checkpoint_and_general_adapter(tmp_path):
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    state = rng.normal(size=(32, 17)).astype(np.float32)
    action = rng.uniform(-1, 1, size=(32, 6)).astype(np.float32)
    next_state = state + 0.01 * rng.normal(size=state.shape).astype(np.float32)
    reward = rng.normal(size=32).astype(np.float32)
    model = MiniDynamics(17, 6, 64)
    model.fit_normalization(state, action, next_state, reward)
    tensors = [torch.from_numpy(x) for x in (state, action, next_state, reward)]
    loss = model.loss(*tensors)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    prediction, predicted_reward = model(tensors[0], tensors[1])
    assert prediction.shape == (32, 17) and predicted_reward.shape == (32,)
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, {"best_epoch": 1})
    restored, metadata = load_checkpoint(path)
    np.testing.assert_allclose(restored(tensors[0], tensors[1])[0].detach(), prediction.detach(), rtol=0, atol=0)
    assert metadata["best_epoch"] == 1
    adapter = LinearResidualAdapter.from_model(restored, ridge=1.0)
    assert adapter.parameter_count == (1 + 17 + 6) * 17
    before = adapter.predict(state[0], action[0]).copy()
    adapter.update(state[0], action[0], np.ones(17) * 0.2)
    assert np.linalg.norm(adapter.predict(state[0], action[0]) - before) > 0
    assert "regime" not in inspect.signature(adapter.update).parameters


def test_cem_uses_learned_model_and_respects_bounds():
    torch.manual_seed(9)
    model = MiniDynamics(17, 6, 64)
    adapter = LinearResidualAdapter(17, 6)
    planner = MiniCEM(model, -np.ones(6), np.ones(6),
                      CEMConfig(horizon=3, candidates=32, iterations=2), seed=9, adapter=adapter)
    sequences = torch.zeros(32, 3, 6)
    states, rewards = planner.rollout(torch.zeros(17), sequences)
    assert states.shape == (32, 3, 17) and rewards.shape == (32, 3)
    action = planner.act(np.zeros(17))
    assert action.shape == (6,) and np.isfinite(action).all()
    assert np.all(action >= -1) and np.all(action <= 1)
    assert planner.last_runtime_seconds > 0


def test_offline_dataset_contains_transitions_only_and_rejects_c(tmp_path):
    path = tmp_path / "train_A.npz"
    _collect_one("A", 8, 8, 123, DEFAULT_REGIMES, path)
    with np.load(path) as data:
        assert set(data.files) == {"state", "action", "next_state", "reward", "terminated", "truncated"}
        assert data["state"].shape == (8, 17)
        assert data["action"].shape == (8, 6)
    assert _load_pooled([path])["state"].shape == (8, 17)
    with pytest.raises(ValueError):
        _load_pooled([tmp_path / "train_C.npz"])
