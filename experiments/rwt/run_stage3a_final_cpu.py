"""Final CPU diagnostic: regime averaging, matched physics, MPC and memory.

C appears only after A/B/D checkpoint selection. Oracle labels are used solely
by the diagnostic oracle-conditioned model and evaluator, never by pooled or
history-conditioned predictors.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

if __package__:
    from .run_stage3a1 import dataset_audit, load_split, model_predict, pool
else:
    from run_stage3a1 import dataset_audit, load_split, model_predict, pool
from rwt.stage3a import (CEMConfig, LinearResidualAdapter, MiniCEM, MiniDynamics,
                         RecurrentHalfCheetah)
from rwt.stage3a1 import ResidualAdapterMemory, TrainNormalizer
from rwt.stage3a_diagnostic import (HISTORY_STEPS, HISTORY_WIDTH,
                                     ConditionedMiniDynamics, TrainSupport, append_history,
                                     observed_histories)


REGIMES = ("A", "B", "D")
METHODS = ("pooled", "pooled_state_only", "oracle", "history")


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def context_for(data_by: dict, method: str) -> np.ndarray | None:
    if method in ("pooled", "pooled_state_only"):
        return None
    if method == "oracle":
        return np.concatenate([np.repeat(np.eye(3, dtype=np.float32)[i:i + 1],
                                         len(data_by[r]["state"]), axis=0)
                               for i, r in enumerate(REGIMES)])
    return np.concatenate([observed_histories(data_by[r]) for r in REGIMES])


def train_one(method: str, seed: int, train: dict, val: dict,
              train_context: np.ndarray | None, val_context: np.ndarray | None,
              config: dict, checkpoint: Path):
    torch.manual_seed(seed)
    model = MiniDynamics(hidden_dim=int(config["hidden_dim"])) if method in ("pooled", "pooled_state_only") else \
        ConditionedMiniDynamics(3 if method == "oracle" else HISTORY_STEPS * HISTORY_WIDTH,
                                hidden_dim=int(config["hidden_dim"]))
    model.fit_normalization(train["state"], train["action"], train["next_state"], train["reward"])
    if train_context is not None:
        model.fit_context_normalization(train_context)
    tensors = lambda data: tuple(torch.as_tensor(data[key], dtype=torch.float32)
                                 for key in ("state", "action", "next_state", "reward"))
    train_t, val_t = tensors(train), tensors(val)
    if train_context is not None:
        train_t += (torch.as_tensor(train_context),)
        val_t += (torch.as_tensor(val_context),)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    best, best_epoch, stale, curve = np.inf, 0, 0, []
    start = perf_counter()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    def objective(batch):
        if method == "pooled_state_only":
            predicted = model(batch[0], batch[1])[0]
            return ((predicted - batch[2]) / model.delta_scale).square().mean()
        return model.loss(*batch)
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train_t[0]))
        total = 0.0
        for indices in permutation.split(int(config["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(tuple(array[indices] for array in train_t))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite diagnostic training loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite diagnostic gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            validation_loss = float(objective(val_t))
        curve.append({"method": method, "seed": seed, "epoch": epoch,
                      "train_loss": total / len(train_t[0]), "validation_loss": validation_loss})
        if validation_loss < best - float(config["min_improvement"]):
            best, best_epoch, stale = validation_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "method": method, "seed": seed,
                        "best_epoch": epoch, "validation_loss": best,
                        "offline_regimes": list(REGIMES)}, checkpoint)
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["offline_regimes"] != list(REGIMES) or saved["method"] != method:
        raise AssertionError("diagnostic checkpoint provenance mismatch")
    model.load_state_dict(saved["model_state"])
    return model.eval(), curve, {"method": method, "seed": seed, "best_epoch": best_epoch,
                                 "epochs_run": len(curve), "validation_loss": best,
                                 "training_seconds": perf_counter() - start,
                                 "parameters": sum(p.numel() for p in model.parameters())}


def predict(model, method: str, state: np.ndarray, action: np.ndarray,
            context: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if method in ("pooled", "pooled_state_only"):
        return model_predict(model, state, action)
    with torch.no_grad():
        next_state, reward = model.predict(torch.as_tensor(state, dtype=torch.float32),
                                           torch.as_tensor(action, dtype=torch.float32),
                                           torch.as_tensor(context, dtype=torch.float32))
    return next_state.numpy(), reward.numpy()


def one_step(models: dict, val_by: dict, normalizer: TrainNormalizer):
    rows, dimensions = [], []
    for regime_index, regime in enumerate(REGIMES):
        data = val_by[regime]
        for (method, seed), model in models.items():
            context = None if method in ("pooled", "pooled_state_only") else (np.repeat(np.eye(3, dtype=np.float32)
                        [regime_index:regime_index + 1], len(data["state"]), axis=0)
                        if method == "oracle" else observed_histories(data))
            prediction, reward = predict(model, method, data["state"], data["action"], context)
            error = (prediction - data["next_state"]) ** 2
            rows.append({"method": method, "seed": seed, "regime": regime,
                         "raw_mse": float(error.mean()),
                         "normalized_mse": normalizer.normalized_mse(prediction, data["next_state"]),
                         "reward_mse": (float(np.mean((reward - data["reward"]) ** 2))
                                        if method != "pooled_state_only" else float("nan"))})
            for dimension in range(17):
                dimensions.append({"method": method, "seed": seed, "regime": regime,
                                   "dimension": dimension, "raw_mse": float(error[:, dimension].mean()),
                                   "normalized_mse": float((error[:, dimension] /
                                                            normalizer.delta_scale[dimension] ** 2).mean())})
    return rows, dimensions


def rollout(models: dict, seeds: list[int]) -> list[dict]:
    rows = []
    for regime in ("A", "B", "D", "C"):
        for seed in seeds:
            env = RecurrentHalfCheetah([(regime, 13)])
            rng = np.random.default_rng(seed + 30)
            try:
                state, _ = env.reset(seed=seed)
                history = np.zeros(HISTORY_STEPS * HISTORY_WIDTH, np.float32)
                for _ in range(3):
                    action = rng.uniform(-0.8, 0.8, 6)
                    next_state, _, _, _, _ = env.step(action)
                    history = append_history(history, state, action, next_state)
                    state = next_state
                actions = rng.uniform(-0.8, 0.8, (10, 6))
                forecasts = {(method, run_seed): (state.copy(), history.copy())
                             for method, run_seed in models if not (regime == "C" and method == "oracle")}
                for horizon, action in enumerate(actions, start=1):
                    for key, (forecast, forecast_history) in list(forecasts.items()):
                        method, run_seed = key
                        context = None if method in ("pooled", "pooled_state_only") else (
                            np.eye(3, dtype=np.float32)[REGIMES.index(regime)][None]
                            if method == "oracle" else forecast_history[None])
                        next_forecast = predict(models[key], method, forecast[None], action[None], context)[0][0]
                        next_history = (append_history(forecast_history, forecast, action, next_forecast)
                                        if method == "history" else forecast_history)
                        forecasts[key] = (next_forecast, next_history)
                    actual, _, _, _, _ = env.step(action)
                    if horizon in (1, 3, 5, 10):
                        for (method, run_seed), (forecast, _) in forecasts.items():
                            rows.append({"regime": regime, "seed": seed, "method": method,
                                         "training_seed": run_seed, "horizon": horizon,
                                         "raw_mse": float(np.mean((forecast - actual) ** 2)),
                                         "finite": bool(np.isfinite(forecast).all())})
            finally:
                env.close()
    return rows


def matched_physics(model: MiniDynamics, state_only: MiniDynamics,
                    oracle: ConditionedMiniDynamics, support: TrainSupport,
                    seed: int = 920) -> list[dict]:
    """Evaluator-only cloned MuJoCo state/action probes, after checkpoint selection."""
    rng = np.random.default_rng(seed)
    source = RecurrentHalfCheetah([("A", 40)])
    snapshots = []
    try:
        source.reset(seed=seed)
        last = np.zeros(6)
        for index in range(32):
            if index < 16:
                action = np.clip(0.55 * last + 0.45 * rng.normal(0, 0.8, 6), -1, 1)
                region = "train_like"
            else:
                action = rng.uniform(-1, 1, 6)
                region = "heldout_action_state"
            qpos = source.unwrapped.data.qpos.copy()
            qvel = source.unwrapped.data.qvel.copy()
            if region == "heldout_action_state":
                qvel *= 1.8
            snapshots.append((region, qpos, qvel, action.copy()))
            source.step(action)
            last = action
    finally:
        source.close()
    rows = []
    for index, (region, qpos, qvel, action) in enumerate(snapshots):
        outputs, errors, state_only_errors, oracle_errors, oracle_predictions = {}, {}, {}, {}, {}
        for regime in ("A", "B", "D", "C"):
            env = RecurrentHalfCheetah([(regime, 1)])
            try:
                env.reset(seed=seed)
                env.unwrapped.set_state(qpos, qvel)
                state = env.unwrapped._get_obs().copy()
                actual, _, _, _, _ = env.step(action)
                predicted = model_predict(model, state[None], action[None])[0][0]
                state_only_predicted = model_predict(state_only, state[None], action[None])[0][0]
                outputs[regime] = actual
                errors[regime] = float(np.linalg.norm(predicted - actual))
                state_only_errors[regime] = float(np.linalg.norm(state_only_predicted - actual))
                if regime in REGIMES:
                    context = np.eye(3, dtype=np.float32)[REGIMES.index(regime)][None]
                    oracle_prediction = predict(oracle, "oracle", state[None], action[None], context)[0][0]
                    oracle_errors[regime] = float(np.linalg.norm(oracle_prediction - actual))
                    oracle_predictions[regime] = oracle_prediction
            finally:
                env.close()
        support_distance = float(support.distance(state[None], action[None])[0])
        closure = outputs["C"] - outputs["B"] - outputs["D"] + outputs["A"]
        denominator = (np.linalg.norm(outputs["B"] - outputs["A"])
                       + np.linalg.norm(outputs["D"] - outputs["A"]))
        for first, second in (("A", "B"), ("A", "D"), ("B", "D"),
                              ("A", "C"), ("B", "C"), ("D", "C")):
            separation = float(np.linalg.norm(outputs[first] - outputs[second]))
            rows.append({"probe": index, "region": region, "first": first, "second": second,
                         "support_distance": support_distance,
                         "outside_val_99pct_support": support_distance > support.threshold,
                         "separation_l2": separation,
                         "pooled_error_l2_first": errors[first],
                         "pooled_error_l2_second": errors[second],
                         "state_only_error_l2_first": state_only_errors[first],
                         "state_only_error_l2_second": state_only_errors[second],
                         "oracle_error_l2_first": oracle_errors.get(first, float("nan")),
                         "oracle_error_l2_second": oracle_errors.get(second, float("nan")),
                         "oracle_predicted_separation_l2": (float(np.linalg.norm(
                             oracle_predictions[first] - oracle_predictions[second]))
                             if first in oracle_predictions and second in oracle_predictions else float("nan")),
                         "separation_to_mean_error": separation /
                         max((errors[first] + errors[second]) / 2, 1e-12),
                         "separation_to_state_only_error": separation /
                         max((state_only_errors[first] + state_only_errors[second]) / 2, 1e-12),
                         "closure_l2": float(np.linalg.norm(closure)),
                         "closure_relative": float(np.linalg.norm(closure) / max(denominator, 1e-12))})
    return rows


def mpc_replay(model: MiniDynamics, support: TrainSupport,
               seeds: list[int]) -> tuple[list[dict], list[dict]]:
    """Compare identical plans from identical qpos/qvel; MuJoCo stays in evaluator."""
    summaries, steps = [], []
    for regime in ("A", "B", "D", "C"):
        for seed in seeds:
            source = RecurrentHalfCheetah([(regime, 10)])
            rng = np.random.default_rng(seed + 731)
            try:
                state, _ = source.reset(seed=seed)
                for _ in range(3):
                    state, _, _, _, _ = source.step(rng.uniform(-0.8, 0.8, 6))
                qpos, qvel = source.unwrapped.data.qpos.copy(), source.unwrapped.data.qvel.copy()
            finally:
                source.close()
            for horizon in (3, 5):
                planner = MiniCEM(model, -np.ones(6), np.ones(6),
                                  CEMConfig(horizon, 32, 2, 0.2), seed=seed, diagnostics=True)
                planner.act(state)
                plans = {"mpc": planner.last_plan_actions,
                         "random": rng.uniform(-1, 1, (horizon, 6)),
                         "zero": np.zeros((horizon, 6))}
                for policy, actions in plans.items():
                    with torch.no_grad():
                        predicted_states, predicted_rewards = planner.rollout(
                            torch.as_tensor(state, dtype=torch.float32),
                            torch.as_tensor(actions[None], dtype=torch.float32))
                    prediction = predicted_states[0].numpy()
                    reward_prediction = predicted_rewards[0].numpy()
                    if policy == "mpc":
                        np.testing.assert_allclose(prediction, planner.last_plan_states, atol=1e-6)
                        np.testing.assert_allclose(reward_prediction, planner.last_plan_rewards, atol=1e-6)
                        np.testing.assert_allclose(reward_prediction.sum(), planner.last_predicted_return, atol=1e-5)
                    replay = RecurrentHalfCheetah([(regime, horizon)])
                    try:
                        replay.reset(seed=seed)
                        replay.unwrapped.set_state(qpos, qvel)
                        np.testing.assert_allclose(replay.unwrapped._get_obs(), state, atol=1e-6)
                        actual_rewards, teacher_forced_rewards = [], []
                        for t, action in enumerate(actions):
                            actual_state = replay.unwrapped._get_obs().copy()
                            teacher_reward = model_predict(model, actual_state[None], action[None])[1][0]
                            actual, reward, _, _, _ = replay.step(action)
                            actual_rewards.append(float(reward))
                            teacher_forced_rewards.append(float(teacher_reward))
                            normalized = (prediction[t] - model.state_mean.numpy()) / model.state_scale.numpy()
                            next_action = actions[t + 1] if t + 1 < horizon else action
                            support_distance = float(support.distance(prediction[t][None],
                                                                      next_action[None])[0])
                            steps.append({"regime": regime, "seed": seed, "horizon": horizon,
                                          "policy": policy, "step": t + 1,
                                          "predicted_reward": float(reward_prediction[t]),
                                          "teacher_forced_reward": float(teacher_reward),
                                          "actual_reward": float(reward),
                                          "state_mse": float(np.mean((prediction[t] - actual) ** 2)),
                                          "predicted_state_ood_6sd": bool((np.abs(normalized) > 6).any()),
                                          "predicted_input_support_distance": support_distance,
                                          "outside_val_99pct_support": support_distance > support.threshold,
                                          "action_norm": float(np.linalg.norm(action))})
                        summaries.append({"regime": regime, "seed": seed, "horizon": horizon,
                                          "policy": policy, "predicted_return": float(reward_prediction.sum()),
                                          "teacher_forced_return": float(sum(teacher_forced_rewards)),
                                          "actual_return": float(sum(actual_rewards)),
                                          "return_gap": float(reward_prediction.sum() - sum(actual_rewards)),
                                          "teacher_forced_gap": float(sum(teacher_forced_rewards) - sum(actual_rewards)),
                                          "planner_seconds": planner.last_runtime_seconds if policy == "mpc" else 0.0,
                                          "invalid_candidates": planner.last_invalid_candidates if policy == "mpc" else 0})
                    finally:
                        replay.close()
    return summaries, steps


def memory_probes(model: MiniDynamics, train: dict, output: Path) -> tuple[list[dict], list[dict]]:
    """Frozen/current/retrieved on identical fixed probes, before any recurrence update."""
    env = RecurrentHalfCheetah([("A", 20), ("C", 20), ("A", 10), ("C", 10)])
    observations = []
    rng = np.random.default_rng(1000)
    try:
        state, _ = env.reset(seed=901)
        for _ in range(60):
            action = rng.uniform(-0.8, 0.8, 6)
            next_state, _, _, _, _ = env.step(action)
            observations.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state
    finally:
        env.close()
    ids = np.random.default_rng(121).choice(len(train["state"]), size=16, replace=False)
    memory = ResidualAdapterMemory(train["state"][ids], train["action"][ids])
    for begin in (0, 20):
        adapter = LinearResidualAdapter.from_model(model, ridge=20.0)
        for state, action, next_state in observations[begin:begin + 20]:
            base = model_predict(model, state[None], action[None])[0][0]
            adapter.update(state, action, next_state - base)
        memory.add(adapter)
    memory.save(output / "diagnostic_memory.npz")
    memory = ResidualAdapterMemory.load(output / "diagnostic_memory.npz")
    retrieval, probes = [], []
    for regime, start, current_index in (("A", 40, 1), ("C", 50, 0)):
        context = observations[start:start + 3]
        selected, context_errors = memory.retrieve(
            np.stack([x[0] for x in context]), np.stack([x[1] for x in context]),
            np.stack([x[2] for x in context]), lambda s, a: model_predict(model, s, a)[0])
        retrieval.append({"regime_evaluator_only": regime, "selected_index": selected,
                          "current_index": current_index,
                          "context_mse_0": float(context_errors[0]),
                          "context_mse_1": float(context_errors[1])})
        probe_env = RecurrentHalfCheetah([(regime, 16)])
        probe_rng = np.random.default_rng(1300 if regime == "A" else 1303)
        try:
            state, _ = probe_env.reset(seed=1300 if regime == "A" else 1303)
            for index in range(16):
                action = probe_rng.uniform(-0.8, 0.8, 6)
                actual, _, _, _, _ = probe_env.step(action)
                base = model_predict(model, state[None], action[None])[0][0]
                for predictor, adapter in (("frozen", None),
                                           ("current", memory.entries[current_index].adapter),
                                           ("retrieved", memory.entries[selected].adapter)):
                    prediction = base if adapter is None else base + adapter.predict(state, action)
                    probes.append({"regime_evaluator_only": regime, "probe": index,
                                   "predictor": predictor,
                                   "mse": float(np.mean((prediction - actual) ** 2))})
                state = actual
        finally:
            probe_env.close()
    return retrieval, probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/rwt/configs/stage3a/final_cpu.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["cpu_threads"]))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config["data_dir"])
    train_by = {r: load_split(data_dir, "train", r) for r in REGIMES}
    val_by = {r: load_split(data_dir, "validation", r) for r in REGIMES}
    audit = dataset_audit(data_dir, train_by, val_by, data_dir.parent / "data_manifest.json")
    train, val = pool(train_by), pool(val_by)
    normalizer = TrainNormalizer.fit(train["state"], train["action"], train["next_state"])
    support = TrainSupport(train["state"], train["action"], val["state"], val["action"])
    models, curves, training = {}, [], []
    for method in METHODS:
        train_context, val_context = context_for(train_by, method), context_for(val_by, method)
        for seed in config["training_seeds"]:
            model, curve, info = train_one(method, int(seed), train, val, train_context, val_context,
                                           config["mlp"], output / "checkpoints" / f"{method}_{seed}.pt")
            models[(method, int(seed))] = model
            curves.extend(curve)
            training.append(info)
    write_csv(output / "training_curves.csv", curves)
    write_csv(output / "training_runs.csv", training)
    one, dimensions = one_step(models, val_by, normalizer)
    write_csv(output / "one_step.csv", one)
    write_csv(output / "per_dimension.csv", dimensions)
    selected = {method: min((row for row in training if row["method"] == method),
                            key=lambda row: row["validation_loss"])["seed"] for method in METHODS}
    selected_models = {(method, selected[method]): models[(method, selected[method])] for method in METHODS}
    # No C simulator is opened before checkpoint selection above.
    roll = rollout(selected_models, [int(x) for x in config["rollout_seeds"]])
    write_csv(output / "rollout.csv", roll)
    pooled = selected_models[("pooled", selected["pooled"])]
    matched = matched_physics(pooled,
                              selected_models[("pooled_state_only", selected["pooled_state_only"])],
                              selected_models[("oracle", selected["oracle"])], support)
    write_csv(output / "matched_physics.csv", matched)
    plans, plan_steps = mpc_replay(pooled, support, [int(x) for x in config["mpc_seeds"]])
    write_csv(output / "mpc_plans.csv", plans)
    write_csv(output / "mpc_plan_steps.csv", plan_steps)
    retrieval, probes = memory_probes(pooled, train, output)
    write_csv(output / "memory_retrieval.csv", retrieval)
    write_csv(output / "memory_fixed_probes.csv", probes)
    summary = {"dataset": audit, "selected_seed_abd_only": selected,
               "val_99pct_nearest_train_input_distance": support.threshold,
               "training_runs": training, "rollout_nonfinite": sum(not row["finite"] for row in roll),
               "mpc_invalid_candidates": sum(row["invalid_candidates"] for row in plans),
               "memory_retrieval": retrieval}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
