import inspect
import json
import hashlib

import numpy as np
import pytest
import torch

from experiments.rwt.run_stage3a import _train
from experiments.rwt.run_stage3a1 import dataset_audit
from rwt.stage3a import (BatchedMiniCEM, CEMConfig, LinearResidualAdapter,
                         MiniCEM, MiniDynamics, load_checkpoint)
from rwt.stage3a1 import LinearRidgeDynamics, ResidualAdapterMemory, TrainNormalizer


def test_dataset_audit_checks_manifest_splits_in_any_order(tmp_path):
    train_by, val_by, splits = {}, {}, []
    for split, target in (("train", train_by), ("validation", val_by)):
        for index, regime in enumerate(("A", "B", "D")):
            value = index + (0 if split == "train" else 10)
            arrays = {"state": np.full((1, 17), value, np.float32),
                      "action": np.zeros((1, 6), np.float32),
                      "next_state": np.full((1, 17), value + 0.1, np.float32),
                      "reward": np.full(1, value, np.float32)}
            target[regime] = arrays
            path = tmp_path / f"{split}_{regime}.npz"
            np.savez(path, **arrays)
            splits.append({"file": str(path), "regime": regime, "count": 1,
                           "seed": value + 100, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"offline_regimes": ["A", "B", "D"], "held_out_regime": "C",
                "splits": list(reversed(splits))}
    path = tmp_path / "data_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert dataset_audit(tmp_path, train_by, val_by, path)["exact_transition_overlap"] == 0
    manifest["splits"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        dataset_audit(tmp_path, train_by, val_by, path)


def test_train_only_normalizer_and_ridge_predict_delta():
    rng = np.random.default_rng(1)
    state = rng.normal(size=(80, 17))
    action = rng.normal(size=(80, 6))
    delta = 0.2 * state[:, :1] + 0.3 * action[:, :1]
    next_state = state + np.repeat(delta, 17, axis=1)
    normalizer = TrainNormalizer.fit(state, action, next_state)
    ridge = LinearRidgeDynamics(normalizer, ridge=1e-3).fit(state, action, next_state)
    assert np.mean((ridge.predict(state, action) - next_state) ** 2) < 1e-6
    shifted = next_state + 100
    assert normalizer.normalized_mse(next_state, shifted) > 1
    np.testing.assert_allclose(normalizer.state_mean, state.mean(axis=0))


def test_adapter_memory_round_trip_and_observation_only_retrieval(tmp_path):
    state = np.zeros(17)
    action = np.zeros(6)
    low = LinearResidualAdapter(17, 6, ridge=1.0)
    high = LinearResidualAdapter(17, 6, ridge=1.0)
    for _ in range(20):
        low.update(state, action, np.full(17, 0.2))
        high.update(state, action, np.full(17, 1.0))
    memory = ResidualAdapterMemory(np.zeros((2, 17)), np.zeros((2, 6)))
    memory.add(low)
    memory.add(high)
    path = tmp_path / "memory.npz"
    memory.save(path)
    restored = ResidualAdapterMemory.load(path)
    selected, errors = restored.retrieve(np.zeros((3, 17)), np.zeros((3, 6)),
                                         np.ones((3, 17)), lambda s, a: np.zeros_like(s))
    assert selected == 1 and errors[1] < errors[0]
    assert len(restored.entries) == 2
    np.testing.assert_array_equal(restored.entries[1].adapter.weights, high.weights)
    assert "regime" not in inspect.signature(restored.retrieve).parameters


def test_batched_cem_cpu_shapes_bounds_and_seed_replay():
    torch.manual_seed(3)
    model = MiniDynamics()
    state = np.zeros((3, 17))
    adapters = [LinearResidualAdapter(17, 6) for _ in range(3)]
    config = CEMConfig(horizon=3, candidates=8, iterations=2)
    first = BatchedMiniCEM(model, -np.ones(6), np.ones(6), config, [11, 12, 13])
    second = BatchedMiniCEM(model, -np.ones(6), np.ones(6), config, [11, 12, 13])
    actions = first.act(state, adapters, [0, 1, 2])
    np.testing.assert_array_equal(actions, second.act(state, adapters, [0, 1, 2]))
    assert actions.shape == (3, 6) and np.all(np.abs(actions) <= 1)
    assert np.isfinite(first.last_predicted_return).all()
    assert np.all((first.last_plan_ood_fraction >= 0) & (first.last_plan_ood_fraction <= 1))


def test_cem_nonfinite_model_falls_back_to_safe_action():
    model = MiniDynamics()
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.fill_(float("nan"))
    planner = MiniCEM(model, -np.ones(6), np.ones(6), CEMConfig(3, 8, 2), seed=1)
    action = planner.act(np.zeros(17))
    np.testing.assert_array_equal(action, np.zeros(6))
    assert planner.last_invalid_candidates == 16


def test_training_resume_reproduces_uninterrupted_cpu_weights(tmp_path):
    rng = np.random.default_rng(4)
    state = rng.normal(size=(64, 17)).astype(np.float32)
    action = rng.uniform(-1, 1, size=(64, 6)).astype(np.float32)
    next_state = state + 0.02 * rng.normal(size=(64, 17)).astype(np.float32)
    reward = rng.normal(size=64).astype(np.float32)
    train = {"state": state, "action": action, "next_state": next_state, "reward": reward}
    config = {"seed": 77, "epochs": 4, "batch_size": 16, "learning_rate": 1e-3}
    torch.manual_seed(77)
    full = MiniDynamics()
    _train(full, train, train, config, "cpu", tmp_path / "full_best.pt",
           training_state=tmp_path / "full_state.pt")
    torch.manual_seed(77)
    partial = MiniDynamics()
    _train(partial, train, train, {**config, "epochs": 2}, "cpu", tmp_path / "partial_best.pt",
           training_state=tmp_path / "partial_state.pt")
    restarted = MiniDynamics()
    _train(restarted, train, train, config, "cpu", tmp_path / "partial_best.pt",
           training_state=tmp_path / "partial_state.pt", resume=True)
    full_loaded, _ = load_checkpoint(tmp_path / "full_best.pt")
    resumed_loaded, _ = load_checkpoint(tmp_path / "partial_best.pt")
    for key, value in full_loaded.state_dict().items():
        torch.testing.assert_close(value, resumed_loaded.state_dict()[key], rtol=0, atol=0)
