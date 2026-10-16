"""Final CPU preflight for the Stage 3A HalfCheetah prototype.

Offline fitting and checkpoint selection use only existing A/B/D data. C is
accessed after model selection, in evaluator-only rollout/control diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from rwt.stage3a import (CEMConfig, DEFAULT_ORDER, LinearResidualAdapter,
                         MiniCEM, MiniDynamics, RecurrentHalfCheetah,
                         load_checkpoint, save_checkpoint)
from rwt.stage3a1 import LinearRidgeDynamics, ResidualAdapterMemory, TrainNormalizer


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_split(data_dir: Path, split: str, regime: str) -> dict[str, np.ndarray]:
    if split not in ("train", "validation") or regime not in ("A", "B", "D"):
        raise ValueError("offline preflight split is restricted to A/B/D")
    with np.load(data_dir / f"{split}_{regime}.npz", allow_pickle=False) as data:
        expected = {"state", "action", "next_state", "reward", "terminated", "truncated"}
        if set(data.files) != expected:
            raise ValueError(f"unexpected dataset fields: {data.files}")
        result = {key: data[key].copy() for key in expected}
    n = len(result["state"])
    if (result["state"].shape != (n, 17) or result["action"].shape != (n, 6)
            or result["next_state"].shape != (n, 17)
            or not all(np.isfinite(result[key]).all() for key in ("state", "action", "next_state", "reward"))):
        raise ValueError("invalid or nonfinite transition array")
    return result


def pool(data: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([data[r][key] for r in ("A", "B", "D")])
            for key in ("state", "action", "next_state", "reward")}


def _row_hashes(data: dict[str, np.ndarray]) -> set[bytes]:
    matrix = np.concatenate((data["state"], data["action"], data["next_state"],
                             data["reward"].reshape(-1, 1)), axis=1)
    return {hashlib.sha256(row.tobytes()).digest() for row in matrix}


def dataset_audit(data_dir: Path, train_by: dict, val_by: dict,
                  manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(path.name for path in data_dir.glob("*.npz"))
    if manifest["offline_regimes"] != ["A", "B", "D"] or manifest["held_out_regime"] != "C":
        raise ValueError("offline manifest violates held-out C boundary")
    if files != sorted(f"{split}_{regime}.npz" for split in ("train", "validation")
                       for regime in ("A", "B", "D")):
        raise ValueError("offline directory contains unexpected files")
    expected = {(split, regime) for split in ("train", "validation") for regime in ("A", "B", "D")}
    recorded = {}
    for item in manifest["splits"]:
        filename = item["file"].replace("\\", "/").rsplit("/", 1)[-1]
        split = "train" if filename.startswith("train_") else "validation" if filename.startswith("validation_") else None
        key = (split, item["regime"])
        if key not in expected or filename != f"{split}_{item['regime']}.npz" or key in recorded:
            raise ValueError("offline manifest contains unexpected or duplicate split")
        actual_count = len((train_by if split == "train" else val_by)[item["regime"]]["state"])
        if int(item["count"]) != actual_count:
            raise ValueError("offline manifest count differs from dataset")
        if "sha256" in item and hashlib.sha256((data_dir / filename).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("offline dataset differs from manifest hash")
        recorded[key] = int(item["seed"])
    if set(recorded) != expected:
        raise ValueError("offline manifest is missing a split")
    train_seeds = {recorded[("train", regime)] for regime in ("A", "B", "D")}
    val_seeds = {recorded[("validation", regime)] for regime in ("A", "B", "D")}
    if train_seeds & val_seeds:
        raise ValueError("train/validation collection seeds overlap")
    train, val = pool(train_by), pool(val_by)
    overlap = len(_row_hashes(train) & _row_hashes(val))
    if overlap:
        raise ValueError("identical transitions appear in train and validation")
    return {"files": files, "train_counts": {r: len(train_by[r]["state"]) for r in train_by},
            "validation_counts": {r: len(val_by[r]["state"]) for r in val_by},
            "train_seeds": sorted(train_seeds), "validation_seeds": sorted(val_seeds),
            "exact_transition_overlap": overlap, "contains_c_offline": False}


def environment_audit() -> dict:
    rng = np.random.default_rng(510)
    action_set = rng.uniform(-0.8, 0.8, (12, 6))
    reference = RecurrentHalfCheetah([("A", 20)])
    snapshots = []
    try:
        reference.reset(seed=510)
        for action in action_set:
            snapshots.append((reference.unwrapped.data.qpos.copy(), reference.unwrapped.data.qvel.copy(),
                              action.copy()))
            reference.step(action)
    finally:
        reference.close()
    predictions = {}
    reset_restored = {}
    for regime in ("A", "B", "D", "C"):
        env = RecurrentHalfCheetah([(regime, 20)])
        try:
            states = []
            for qpos, qvel, action in snapshots:
                env.reset(seed=510)
                env.unwrapped.set_state(qpos, qvel)
                next_state, _, _, _, info = env.step(action)
                if "regime" in info or "change" in info or "physics" in info:
                    raise AssertionError("environment leaked change metadata in info")
                states.append(next_state)
            predictions[regime] = np.stack(states)
            env.reset(seed=510)
            expected_friction = env._base_friction * env.regimes[regime].friction
            expected_gain = env._base_gain.copy()
            expected_gain[:, 0] *= env.regimes[regime].actuator_gain
            reset_restored[regime] = bool(np.array_equal(env.unwrapped.model.geom_friction, expected_friction)
                                          and np.array_equal(env.unwrapped.model.actuator_gainprm, expected_gain))
        finally:
            env.close()
    pairwise = {f"{a}-{b}": float(np.mean(np.linalg.norm(predictions[a] - predictions[b], axis=1)))
                for a, b in (("A", "B"), ("A", "D"), ("A", "C"), ("B", "C"), ("D", "C"))}
    if not all(value > 1e-6 for value in pairwise.values()) or not all(reset_restored.values()):
        raise AssertionError("regime perturbation or reset restoration failed")
    replay = []
    for _ in range(2):
        env = RecurrentHalfCheetah([("A", 2), ("B", 2), ("D", 2), ("C", 2), ("A", 2), ("C", 2)])
        try:
            state, info = env.reset(seed=511)
            if "regime" in info:
                raise AssertionError("reset info leaked regime")
            trace = [state.copy()]
            labels = []
            for action in action_set:
                state, _, _, _, info = env.step(action)
                labels.append(env.last_transition_regime)
                trace.append(state.copy())
            replay.append((np.stack(trace), labels))
        finally:
            env.close()
    deterministic = bool(np.array_equal(replay[0][0], replay[1][0]))
    schedule_ok = replay[0][1] == [x for x in DEFAULT_ORDER for _ in range(2)]
    if not deterministic or not schedule_ok:
        raise AssertionError("seed replay or schedule switching failed")
    return {"pairwise_next_state_l2": pairwise, "reset_restored": reset_restored,
            "deterministic_replay": deterministic, "schedule_ok": schedule_ok,
            "agent_observation_shape": [17], "agent_action_shape": [6]}


def train_mlp(train: dict, validation: dict, seed: int, config: dict, path: Path):
    torch.manual_seed(seed)
    model = MiniDynamics(17, 6, int(config["hidden_dim"]))
    model.fit_normalization(train["state"], train["action"], train["next_state"], train["reward"])
    tensors = lambda data: tuple(torch.as_tensor(data[key], dtype=torch.float32) for key in
                                 ("state", "action", "next_state", "reward"))
    train_t, val_t = tensors(train), tensors(validation)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    best, best_epoch, stale = np.inf, 0, 0
    history = []
    start = perf_counter()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train_t[0]))
        train_total = 0.0
        for indices in permutation.split(int(config["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(*(array[indices] for array in train_t))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite MLP loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite MLP gradient")
            optimizer.step()
            train_total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            val_loss = float(model.loss(*val_t))
        history.append({"seed": seed, "epoch": epoch, "train_loss": train_total / len(train_t[0]),
                        "validation_loss": val_loss})
        if val_loss < best - float(config["min_improvement"]):
            best, best_epoch, stale = val_loss, epoch, 0
            save_checkpoint(model, path, {"seed": seed, "best_epoch": epoch,
                                          "validation_loss": best, "offline_regimes": ["A", "B", "D"]})
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    restored, meta = load_checkpoint(path)
    if meta["offline_regimes"] != ["A", "B", "D"]:
        raise AssertionError("checkpoint metadata contaminated by C")
    expected = TrainNormalizer.fit(train["state"], train["action"], train["next_state"])
    for name in ("state_mean", "state_scale", "action_mean", "action_scale", "delta_mean", "delta_scale"):
        np.testing.assert_allclose(getattr(restored, name).numpy(), getattr(expected, name), rtol=2e-4, atol=1e-4)
    train_reward = torch.as_tensor(train["reward"], dtype=torch.float32)
    np.testing.assert_allclose(restored.reward_mean.numpy(), train_reward.mean().numpy(), atol=1e-6)
    np.testing.assert_allclose(restored.reward_scale.numpy(),
                               train_reward.std(unbiased=False).clamp_min(1e-4).numpy(), atol=1e-6)
    return restored, history, {"seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
                               "best_validation_loss": best, "training_seconds": perf_counter() - start}


def model_predict(model: MiniDynamics, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    with torch.no_grad():
        next_state, reward = model.predict(torch.as_tensor(state, dtype=torch.float32, device=device),
                                           torch.as_tensor(action, dtype=torch.float32, device=device))
    return next_state.detach().cpu().numpy(), reward.detach().cpu().numpy()


def validation_metrics(normalizer: TrainNormalizer, val_by: dict, ridge: LinearRidgeDynamics,
                       mlps: dict[int, MiniDynamics]):
    summaries, dimensions, rewards = [], [], []
    for regime, data in val_by.items():
        target = data["next_state"]
        predictions = {"persistence": data["state"],
                       "linear_ridge": ridge.predict(data["state"], data["action"])}
        for seed, model in mlps.items():
            predictions[f"mlp_{seed}"] = model_predict(model, data["state"], data["action"])[0]
            reward_pred = model_predict(model, data["state"], data["action"])[1]
            rewards.append({"regime": regime, "seed": seed,
                            "reward_mse": float(np.mean((reward_pred - data["reward"]) ** 2)),
                            "reward_mae": float(np.mean(np.abs(reward_pred - data["reward"])))})
        for method, prediction in predictions.items():
            squared = (prediction - target) ** 2
            summaries.append({"regime": regime, "method": method, "count": len(target),
                              "raw_mse": float(squared.mean()),
                              "normalized_mse": normalizer.normalized_mse(prediction, target)})
            for dim in range(target.shape[1]):
                dimensions.append({"regime": regime, "method": method, "dimension": dim,
                                   "raw_mse": float(squared[:, dim].mean()),
                                   "normalized_mse": float((squared[:, dim] /
                                                           normalizer.delta_scale[dim] ** 2).mean())})
    return summaries, dimensions, rewards


def rollout_metrics(mlp: MiniDynamics, ridge: LinearRidgeDynamics, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        anchor = RecurrentHalfCheetah([("A", 10)])
        try:
            anchor.reset(seed=seed)
            qpos, qvel = anchor.unwrapped.data.qpos.copy(), anchor.unwrapped.data.qvel.copy()
        finally:
            anchor.close()
        actions = np.random.default_rng(seed + 991).uniform(-0.8, 0.8, size=(10, 6))
        for regime in ("A", "B", "D", "C"):
            env = RecurrentHalfCheetah([(regime, 10)])
            try:
                env.reset(seed=seed)
                env.unwrapped.set_state(qpos, qvel)
                initial = env.unwrapped._get_obs().copy()
                predicted = {"persistence": initial.copy(), "linear_ridge": initial.copy(),
                             "mlp": initial.copy()}
                for step, action in enumerate(actions, start=1):
                    for method in predicted:
                        state = predicted[method]
                        if method == "linear_ridge":
                            predicted[method] = ridge.predict(state, action)
                        elif method == "mlp":
                            predicted[method] = model_predict(mlp, state[None], action[None])[0][0]
                    actual, _, _, _, _ = env.step(action)
                    if step in (1, 3, 5, 10):
                        for method, value in predicted.items():
                            finite = bool(np.isfinite(value).all())
                            rows.append({"seed": seed, "regime": regime, "horizon": step,
                                         "method": method,
                                         "raw_mse": float(np.mean((value - actual) ** 2)) if finite else float("inf"),
                                         "finite": finite, "max_abs_prediction": float(np.max(np.abs(value)))})
            finally:
                env.close()
    return rows


def mpc_metrics(model: MiniDynamics, seeds: list[int], mpc_config: dict) -> tuple[list[dict], list[dict]]:
    schedule = list(zip(DEFAULT_ORDER, [8, 8, 8, 8, 8, 8]))
    episodes, steps = [], []
    for seed in seeds:
        for policy in ("zero", "random", "mpc"):
            env = RecurrentHalfCheetah(schedule)
            rng = np.random.default_rng(seed + 333)
            planner = MiniCEM(model, env.action_space.low, env.action_space.high,
                              CEMConfig(**mpc_config), seed=seed + 333, diagnostics=True)
            actual_total = predicted_total = 0.0
            action_norms, times, ood, invalid = [], [], [], []
            try:
                state, _ = env.reset(seed=seed)
                for t in range(48):
                    if policy == "zero":
                        action = np.zeros(6)
                    elif policy == "random":
                        action = rng.uniform(-1, 1, 6)
                    else:
                        action = planner.act(state)
                    if not env.action_space.contains(action.astype(env.action_space.dtype)):
                        raise AssertionError("policy action out of bounds")
                    predicted_next, predicted_reward = model_predict(model, state[None], action[None])
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    error = float(np.mean((predicted_next[0] - next_state) ** 2))
                    steps.append({"seed": seed, "policy": policy, "step": t,
                                  "regime": env.last_transition_regime, "predicted_reward": float(predicted_reward[0]),
                                  "actual_reward": float(reward), "next_state_mse": error,
                                  "action_norm": float(np.linalg.norm(action)),
                                  "mpc_seconds": planner.last_runtime_seconds if policy == "mpc" else 0.0,
                                  "planned_horizon_return": planner.last_predicted_return if policy == "mpc" else float("nan"),
                                  "plan_ood_fraction": planner.last_plan_ood_fraction if policy == "mpc" else float("nan"),
                                  "invalid_candidates": planner.last_invalid_candidates if policy == "mpc" else 0})
                    actual_total += float(reward)
                    predicted_total += float(predicted_reward[0])
                    action_norms.append(float(np.linalg.norm(action)))
                    if policy == "mpc":
                        times.append(planner.last_runtime_seconds)
                        ood.append(planner.last_plan_ood_fraction)
                        invalid.append(planner.last_invalid_candidates)
                    state = next_state
                    if terminated or truncated:
                        break
                policy_steps = [row for row in steps if row["seed"] == seed and row["policy"] == policy]
                episodes.append({"seed": seed, "policy": policy, "steps": len(policy_steps),
                                 "predicted_return_teacher_forced": predicted_total,
                                 "actual_return": actual_total,
                                 "return_gap": predicted_total - actual_total,
                                 "next_state_mse": float(np.mean([x["next_state_mse"] for x in policy_steps])),
                                 "action_norm": float(np.mean(action_norms)),
                                 "mpc_mean_seconds": float(np.mean(times)) if times else 0.0,
                                 "max_plan_ood_fraction": float(np.max(ood)) if ood else 0.0,
                                 "invalid_candidates": int(sum(invalid))})
            finally:
                env.close()
    return episodes, steps


def memory_diagnostic(model: MiniDynamics, train: dict, output: Path, seed: int = 901) -> tuple[list[dict], dict]:
    schedule = [("A", 20), ("C", 20), ("A", 10), ("C", 10)]
    env = RecurrentHalfCheetah(schedule)
    observations = []
    rng = np.random.default_rng(seed + 99)
    try:
        state, _ = env.reset(seed=seed)
        for _ in range(60):
            action = rng.uniform(-0.8, 0.8, 6)
            next_state, _, _, _, _ = env.step(action)
            observations.append((state.copy(), action.copy(), next_state.copy(), env.last_transition_regime))
            state = next_state
    finally:
        env.close()
    probe_ids = np.random.default_rng(121).choice(len(train["state"]), size=16, replace=False)
    memory = ResidualAdapterMemory(train["state"][probe_ids], train["action"][probe_ids])
    for begin, end in ((0, 20), (20, 40)):
        adapter = LinearResidualAdapter.from_model(model, ridge=20.0)
        for state, action, next_state, _ in observations[begin:end]:
            base = model_predict(model, state[None], action[None])[0][0]
            if not np.isfinite(adapter.covariance).all():
                raise FloatingPointError("nonfinite RLS covariance")
            adapter.update(state, action, next_state - base)
        if (not np.isfinite(adapter.weights).all() or not np.isfinite(adapter.covariance).all()
                or np.linalg.eigvalsh(adapter.covariance).min() < -1e-9):
            raise FloatingPointError("RLS update became numerically unstable")
        memory.add(adapter)
    path = output / "adapter_memory.npz"
    memory.save(path)
    restored = ResidualAdapterMemory.load(path)
    signature_distance = float(np.linalg.norm(restored.entries[0].signature - restored.entries[1].signature))
    rows = []
    for regime, start in (("A", 40), ("C", 50)):
        context = observations[start:start + 3]
        context_s = np.stack([x[0] for x in context])
        context_a = np.stack([x[1] for x in context])
        context_n = np.stack([x[2] for x in context])
        base_predict = lambda s, a: model_predict(model, s, a)[0]
        selected, context_errors = restored.retrieve(context_s, context_a, context_n, base_predict)
        for offset, (state, action, next_state, _) in enumerate(observations[start + 3:start + 8], start=3):
            base = model_predict(model, state[None], action[None])[0][0]
            corrected = base + restored.entries[selected].adapter.predict(state, action)
            rows.append({"regime_evaluator_only": regime, "offset": offset,
                         "selected_memory_index": selected, "correct_index_evaluator_only": 0 if regime == "A" else 1,
                         "context_mse_entry_0": float(context_errors[0]),
                         "context_mse_entry_1": float(context_errors[1]),
                         "fresh_mse_before_update": float(np.mean((base - next_state) ** 2)),
                         "retrieved_mse_before_update": float(np.mean((corrected - next_state) ** 2))})
    return rows, {"memory_entries": len(restored.entries), "signature_l2_distance": signature_distance,
                  "retrieval_accuracy": float(np.mean([x["selected_memory_index"] == x["correct_index_evaluator_only"]
                                                      for x in rows])),
                  "fresh_mse": float(np.mean([x["fresh_mse_before_update"] for x in rows])),
                  "retrieved_mse": float(np.mean([x["retrieved_mse_before_update"] for x in rows]))}


def fixed_regime_probes(model: MiniDynamics, memory_path: Path) -> list[dict]:
    """Evaluator-only probe sets; never used for fit, selection or retrieval."""
    memory = ResidualAdapterMemory.load(memory_path)
    rows = []
    for index, regime in enumerate(("A", "B", "D", "C")):
        env = RecurrentHalfCheetah([(regime, 16)])
        rng = np.random.default_rng(1300 + index)
        try:
            state, _ = env.reset(seed=1300 + index)
            for step in range(16):
                action = rng.uniform(-0.8, 0.8, 6)
                base = model_predict(model, state[None], action[None])[0][0]
                next_state, _, _, _, _ = env.step(action)
                rows.append({"regime_evaluator_only": regime, "probe": step,
                             "predictor": "frozen_mlp", "mse": float(np.mean((base - next_state) ** 2))})
                for memory_index, entry in enumerate(memory.entries):
                    revised = base + entry.adapter.predict(state, action)
                    rows.append({"regime_evaluator_only": regime, "probe": step,
                                 "predictor": f"memory_{memory_index}",
                                 "mse": float(np.mean((revised - next_state) ** 2))})
                state = next_state
        finally:
            env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/rwt/configs/stage3a/preflight_cpu.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(int(config.get("cpu_threads", 2)))
    data_dir, output = Path(config["data_dir"]), Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    train_by = {r: load_split(data_dir, "train", r) for r in ("A", "B", "D")}
    val_by = {r: load_split(data_dir, "validation", r) for r in ("A", "B", "D")}
    train, validation = pool(train_by), pool(val_by)
    audit = dataset_audit(data_dir, train_by, val_by, data_dir.parent / "data_manifest.json")
    normalizer = TrainNormalizer.fit(train["state"], train["action"], train["next_state"])
    ridge = LinearRidgeDynamics(normalizer, ridge=float(config["ridge"])).fit(
        train["state"], train["action"], train["next_state"])
    mlps, histories, training_rows = {}, [], []
    for seed in config["training_seeds"]:
        model, history, info = train_mlp(train, validation, int(seed), config["mlp"],
                                         output / "checkpoints" / f"mlp_seed_{seed}.pt")
        mlps[int(seed)] = model
        histories += history
        training_rows.append(info)
    write_csv(output / "training_curves.csv", histories)
    write_csv(output / "training_seeds.csv", training_rows)
    one_step, dimensions, rewards = validation_metrics(normalizer, val_by, ridge, mlps)
    write_csv(output / "one_step.csv", one_step)
    write_csv(output / "per_dimension.csv", dimensions)
    write_csv(output / "reward_validation.csv", rewards)
    selected_seed = min(training_rows, key=lambda row: row["best_validation_loss"])["seed"]
    selected_model = mlps[selected_seed]
    # C is first accessed only after the A/B/D-only checkpoint is fixed.
    environment = environment_audit()
    rollout = rollout_metrics(selected_model, ridge, [int(x) for x in config["rollout_seeds"]])
    write_csv(output / "rollout.csv", rollout)
    mpc_episodes, mpc_steps = mpc_metrics(selected_model, [int(x) for x in config["mpc_seeds"]], config["mpc"])
    write_csv(output / "mpc_episodes.csv", mpc_episodes)
    write_csv(output / "mpc_steps.csv", mpc_steps)
    memory_rows, memory_summary = memory_diagnostic(selected_model, train, output)
    write_csv(output / "memory_retrieval.csv", memory_rows)
    probe_rows = fixed_regime_probes(selected_model, output / "adapter_memory.npz")
    write_csv(output / "regime_probes.csv", probe_rows)
    summary = {"dataset": audit, "environment": environment,
               "selected_seed_by_abd_validation": selected_seed,
               "training": training_rows,
               "rollout_nonfinite": sum(not x["finite"] for x in rollout),
               "rollout_max_abs_prediction": max(x["max_abs_prediction"] for x in rollout),
               "memory": memory_summary,
               "mpc_any_ood_plan": sum(x["max_plan_ood_fraction"] > 0 for x in mpc_episodes),
               "mpc_invalid_candidates": sum(x["invalid_candidates"] for x in mpc_episodes)}
    warnings = []
    if summary["mpc_any_ood_plan"]:
        warnings.append("MPC selected at least one predicted trajectory outside 6 train-state SD in one or more dimensions")
    if summary["mpc_invalid_candidates"]:
        warnings.append("MPC generated nonfinite candidate predictions; invalid values were rejected")
    summary["warnings"] = warnings
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
