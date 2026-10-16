import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.rwt.collect_stage3b_coverage import collect_regime
from experiments.rwt.run_stage3b_pilot import _train, load_train_addendum
from rwt.stage3a import CEMConfig
from rwt.stage3b import RewardPredictor, SeparatedWorldModel, StateDynamics


def test_separate_state_and_reward_gradients_and_shapes():
    dynamics, reward = StateDynamics(), RewardPredictor()
    state, action = torch.randn(4, 17), torch.randn(4, 6)
    dynamics.loss(state, action, state).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dynamics.parameters())
    assert all(p.grad is None for p in reward.parameters())
    dynamics.zero_grad(set_to_none=True)
    reward.loss(state, action, torch.zeros(4)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reward.parameters())
    assert all(p.grad is None for p in dynamics.parameters())
    next_state, predicted_reward = SeparatedWorldModel(dynamics, reward)(state, action)
    assert next_state.shape == (4, 17) and predicted_reward.shape == (4,)


def test_context_required_for_conditioned_dynamics():
    model = StateDynamics(context_dim=3)
    state, action = torch.zeros((2, 17)), torch.zeros((2, 6))
    with pytest.raises(ValueError, match="context"):
        model(state, action)
    assert model(state, action, torch.eye(3)[:2]).shape == (2, 17)
    with pytest.raises(ValueError, match="pooled"):
        StateDynamics()(state, action, torch.eye(3)[:2])


def test_state_checkpoint_resume_exact_cpu_and_train_only_normalization(tmp_path):
    rng = np.random.default_rng(5)
    state = rng.normal(size=(64, 17)).astype(np.float32)
    action = rng.uniform(-1, 1, size=(64, 6)).astype(np.float32)
    next_state = state + rng.normal(0, .1, size=(64, 17)).astype(np.float32)
    reward = rng.normal(size=64).astype(np.float32)
    train = {"state": state, "action": action, "next_state": next_state, "reward": reward}
    val = {**train, "next_state": next_state + 100}
    settings = {"hidden_dim": 16, "epochs": 3, "batch_size": 16, "learning_rate": 1e-3}
    full, _, _ = _train("pooled", 51, train, val, None, None, settings,
                        tmp_path / "full", "cpu", False)
    _train("pooled", 51, train, val, None, None, {**settings, "epochs": 1},
           tmp_path / "partial", "cpu", False)
    resumed, _, _ = _train("pooled", 51, train, val, None, None, settings,
                           tmp_path / "partial", "cpu", True)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(full.delta_mean,
                               torch.as_tensor(next_state - state).mean(0))


def test_reward_checkpoint_uses_independent_train_statistics(tmp_path):
    rng = np.random.default_rng(12)
    train = {"state": rng.normal(size=(32, 17)).astype(np.float32),
             "action": rng.normal(size=(32, 6)).astype(np.float32),
             "next_state": rng.normal(size=(32, 17)).astype(np.float32),
             "reward": rng.normal(size=32).astype(np.float32)}
    validation = {**train, "reward": train["reward"] + 100}
    reward_model, _, _ = _train("reward", 9, train, validation, None, None,
                                {"hidden_dim": 16, "epochs": 2, "batch_size": 8,
                                 "learning_rate": 1e-3}, tmp_path, "cpu", False)
    torch.testing.assert_close(reward_model.reward_mean,
                               torch.as_tensor(train["reward"]).mean().reshape(1))
    assert reward_model(torch.zeros(2, 17), torch.zeros(2, 6)).shape == (2,)


def test_addendum_loader_checks_hash_and_excludes_c(tmp_path):
    rows = []
    for index, regime in enumerate(("A", "B", "D")):
        path = tmp_path / f"train_extra_{regime}.npz"
        np.savez(path, state=np.zeros((1, 17), np.float32),
                 action=np.zeros((1, 6), np.float32),
                 next_state=np.zeros((1, 17), np.float32), reward=np.zeros(1, np.float32),
                 terminated=np.zeros(1, bool), truncated=np.ones(1, bool))
        rows.append({"regime_evaluator_only": regime, "file": str(path),
                     "count": 1, "seed": 100 + index,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"offline_regimes": ["A", "B", "D"], "held_out_regime": "C",
                "split": "train_only", "splits": rows}
    path = tmp_path / "coverage_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    data, audit = load_train_addendum(tmp_path, {1, 2, 3})
    assert set(data) == {"A", "B", "D"} and audit["validation_unchanged"]
    manifest["splits"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_train_addendum(tmp_path, {1, 2, 3})


def test_coverage_collector_is_bounded_and_train_only():
    world = SeparatedWorldModel(StateDynamics(), RewardPredictor()).eval()
    data, info = collect_regime("A", 901, 1, 1, world, CEMConfig(2, 8, 1, .25))
    assert len(data["state"]) == 2 and data["action"].shape == (2, 6)
    assert np.all(np.abs(data["action"]) <= 1) and data["truncated"].all()
    assert set(data) == {"state", "action", "next_state", "reward", "terminated", "truncated"}
    assert info["regime_evaluator_only"] == "A"
    with pytest.raises(ValueError, match="A/B/D"):
        collect_regime("C", 901, 1, 1, world, CEMConfig(2, 8, 1, .25))
