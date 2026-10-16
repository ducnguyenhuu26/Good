"""Stage 3B preparation: fair state-only baselines and separate reward pilot.

Run from the repository root with PYTHONPATH=src. C is opened only after all
A/B/D checkpoints, normalizers and validation-only selections are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

if __package__:
    from .run_stage3a1 import _row_hashes, dataset_audit, load_split, pool
    from .run_stage3a_final_cpu import context_for, mpc_replay, write_csv
else:
    from run_stage3a1 import _row_hashes, dataset_audit, load_split, pool
    from run_stage3a_final_cpu import context_for, mpc_replay, write_csv
from rwt.stage3a import RecurrentHalfCheetah
from rwt.stage3a1 import LinearRidgeDynamics, TrainNormalizer
from rwt.stage3a_diagnostic import (HISTORY_STEPS, HISTORY_WIDTH,
                                     TrainSupport, append_history,
                                     observed_histories)
from rwt.stage3b import RewardPredictor, SeparatedWorldModel, StateDynamics


REGIMES = ("A", "B", "D")
METHODS = ("pooled", "oracle", "history")


def load_train_addendum(directory: Path, forbidden_seeds: set[int]) -> tuple[dict, dict]:
    manifest = json.loads((directory / "coverage_manifest.json").read_text(encoding="utf-8"))
    if (manifest["offline_regimes"] != list(REGIMES) or manifest["held_out_regime"] != "C"
            or manifest["split"] != "train_only" or len(manifest["splits"]) != 3):
        raise ValueError("coverage addendum violates A/B/D train-only boundary")
    if sorted(path.name for path in directory.glob("*.npz")) != [f"train_extra_{r}.npz" for r in REGIMES]:
        raise ValueError("coverage addendum has unexpected NPZ files")
    data, seeds = {}, set()
    expected_keys = {"state", "action", "next_state", "reward", "terminated", "truncated"}
    for item in manifest["splits"]:
        regime = item["regime_evaluator_only"]
        if regime not in REGIMES or regime in data:
            raise ValueError("coverage addendum regime duplicated or unknown")
        path = directory / f"train_extra_{regime}.npz"
        if Path(item["file"]).name != path.name or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("coverage addendum checksum or name mismatch")
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != expected_keys:
                raise ValueError("coverage addendum has unexpected dataset fields")
            arrays = {key: source[key].copy() for key in expected_keys}
        count = len(arrays["state"])
        if (count != int(item["count"]) or arrays["state"].shape != (count, 17)
                or arrays["action"].shape != (count, 6)
                or arrays["next_state"].shape != (count, 17)
                or not all(np.isfinite(arrays[k]).all() for k in ("state", "action", "next_state", "reward"))):
            raise ValueError("coverage addendum has invalid transition arrays")
        data[regime] = arrays
        seeds.add(int(item["seed"]))
    if set(data) != set(REGIMES) or seeds & forbidden_seeds or len(seeds) != 3:
        raise ValueError("coverage addendum overlaps split seeds or misses a regime")
    return data, {"extra_train_counts": {r: len(data[r]["state"]) for r in REGIMES},
                  "extra_train_seeds": sorted(seeds), "validation_unchanged": True}


def _model(method: str, hidden_dim: int) -> StateDynamics:
    return StateDynamics(0 if method == "pooled" else 3 if method == "oracle"
                         else HISTORY_STEPS * HISTORY_WIDTH, hidden_dim)


def _arrays(data: dict, device: str, context: np.ndarray | None = None):
    arrays = tuple(torch.as_tensor(data[name], dtype=torch.float32, device=device)
                   for name in ("state", "action", "next_state", "reward"))
    return arrays + ((torch.as_tensor(context, dtype=torch.float32, device=device),)
                     if context is not None else ())


def _train(kind: str, seed: int, train: dict, val: dict,
           train_context: np.ndarray | None, val_context: np.ndarray | None,
           settings: dict, output: Path, device: str, resume: bool):
    if device == "cuda":
        # Independent GPU workers should not each start a large CPU thread pool.
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    hidden = int(settings["hidden_dim"])
    model = RewardPredictor(hidden) if kind == "reward" else _model(kind, hidden)
    model.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_path = output / "checkpoints" / f"{kind}_{seed}_best.pt"
    latest_path = output / "checkpoints" / f"{kind}_{seed}_latest.pt"
    if resume and not best_path.exists() and not latest_path.exists():
        resume = False
    elif resume and (not best_path.exists() or not latest_path.exists()):
        raise FileNotFoundError(f"incomplete resume checkpoint pair for {kind}/{seed}")
    if not resume:
        if kind == "reward":
            model.fit_normalization(train["state"], train["action"], train["reward"])
        else:
            model.fit_normalization(train["state"], train["action"], train["next_state"], train_context)
    train_t, val_t = _arrays(train, device, train_context), _arrays(val, device, val_context)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best, start_epoch, curve = np.inf, 1, []
    if resume:
        if not best_path.exists() or not latest_path.exists():
            raise FileNotFoundError(f"resume requires both checkpoints for {kind}/{seed}")
        saved = torch.load(latest_path, map_location=device, weights_only=True)
        if saved["kind"] != kind or int(saved["seed"]) != seed or saved["offline_regimes"] != list(REGIMES):
            raise ValueError("resume checkpoint provenance mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        best, start_epoch, curve = float(saved["best_validation_loss"]), int(saved["epoch"]) + 1, saved["curve"]
        torch.set_rng_state(saved["cpu_rng_state"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([item.cpu() for item in saved["cuda_rng_state"]])

    def loss(batch):
        if kind == "reward":
            return model.loss(batch[0], batch[1], batch[3])
        return model.loss(batch[0], batch[1], batch[2], batch[4] if len(batch) == 5 else None)

    start = perf_counter()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_t[0]), device=device)
        total = torch.zeros((), dtype=torch.float32, device=device)
        finite_losses = torch.ones((), dtype=torch.bool, device=device)
        for indices in order.split(int(settings["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss(tuple(array[indices] for array in train_t))
            finite_losses &= torch.isfinite(batch_loss.detach())
            batch_loss.backward()
            optimizer.step()
            total += batch_loss.detach() * len(indices)
        model.eval()
        with torch.no_grad():
            validation_loss_tensor = loss(val_t)
        finite_parameters = torch.stack(
            [torch.isfinite(parameter).all() for parameter in model.parameters()]
        ).all()
        epoch_stats = torch.stack((
            total / len(train_t[0]),
            validation_loss_tensor,
            (finite_losses & torch.isfinite(validation_loss_tensor) & finite_parameters).to(total.dtype),
        )).detach().cpu().tolist()
        train_loss, validation_loss, finite_epoch = epoch_stats
        if not finite_epoch:
            raise FloatingPointError("nonfinite pilot model or validation loss")
        curve.append({"kind": kind, "seed": seed, "epoch": epoch,
                      "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best:
            best = validation_loss
            torch.save({"kind": kind, "seed": seed, "epoch": epoch,
                        "validation_loss": best, "offline_regimes": list(REGIMES),
                        "model_state": model.state_dict()}, best_path)
        latest = {"kind": kind, "seed": seed, "epoch": epoch, "best_validation_loss": best,
                  "offline_regimes": list(REGIMES), "model_state": model.state_dict(),
                  "optimizer_state": optimizer.state_dict(), "cpu_rng_state": torch.get_rng_state(),
                  "curve": curve}
        if device == "cuda":
            latest["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        temporary = latest_path.with_suffix(".tmp")
        torch.save(latest, temporary)
        temporary.replace(latest_path)
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    loaded = RewardPredictor(hidden) if kind == "reward" else _model(kind, hidden)
    loaded.load_state_dict(selected["model_state"])
    loaded.eval()
    train_values = {"state": train["state"], "action": train["action"]}
    if kind == "reward":
        train_values["reward"] = train["reward"].reshape(-1, 1)
    else:
        train_values["delta"] = train["next_state"] - train["state"]
        if train_context is not None:
            train_values["context"] = train_context
    for name, value in train_values.items():
        tensor = torch.as_tensor(value, dtype=torch.float32)
        torch.testing.assert_close(getattr(loaded, f"{name}_mean"), tensor.mean(0))
        torch.testing.assert_close(getattr(loaded, f"{name}_scale"),
                                   tensor.std(0, unbiased=False).clamp_min(1e-4))
    row = {"kind": kind, "seed": seed,
           "best_epoch": int(selected["epoch"]), "validation_loss": best,
           "training_seconds_this_invocation": perf_counter() - start,
           "parameters": sum(p.numel() for p in loaded.parameters())}
    if device == "cuda":
        row["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        row["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
    return loaded, curve, row


def predict_state(model: StateDynamics, state: np.ndarray, action: np.ndarray,
                  context: np.ndarray | None = None) -> np.ndarray:
    with torch.no_grad():
        prediction = model(torch.as_tensor(state, dtype=torch.float32),
                           torch.as_tensor(action, dtype=torch.float32),
                           torch.as_tensor(context, dtype=torch.float32) if context is not None else None)
    return prediction.numpy()


def validation_metrics(models: dict, reward_model: RewardPredictor,
                       val_by: dict, normalizer: TrainNormalizer):
    rows, dimensions, rewards = [], [], []
    for regime_index, regime in enumerate(REGIMES):
        data = val_by[regime]
        for (kind, seed), model in models.items():
            context = None if kind == "pooled" else (
                np.repeat(np.eye(3, dtype=np.float32)[regime_index:regime_index + 1],
                          len(data["state"]), axis=0) if kind == "oracle" else
                observed_histories(data))
            prediction = predict_state(model, data["state"], data["action"], context)
            squared = (prediction - data["next_state"]) ** 2
            rows.append({"method": kind, "seed": seed, "eval_seed": -1, "regime": regime,
                         "split": "validation_abd",
                         "raw_mse": float(squared.mean()),
                         "normalized_mse": normalizer.normalized_mse(prediction, data["next_state"])})
            for dimension in range(17):
                dimensions.append({"method": kind, "seed": seed, "eval_seed": -1, "regime": regime,
                                   "split": "validation_abd",
                                   "dimension": dimension, "raw_mse": float(squared[:, dimension].mean()),
                                   "normalized_mse": float((squared[:, dimension] /
                                                            normalizer.delta_scale[dimension] ** 2).mean())})
        with torch.no_grad():
            prediction = reward_model(torch.as_tensor(data["state"]),
                                      torch.as_tensor(data["action"])).numpy()
        rewards.append({"regime": regime, "seed": -1, "eval_seed": -1, "split": "validation_abd",
                        "reward_mse": float(np.mean((prediction - data["reward"]) ** 2))})
    return rows, dimensions, rewards


def heldout_metrics(models: dict, ridge: LinearRidgeDynamics, reward_model: RewardPredictor,
                    normalizer: TrainNormalizer, seeds: list[int], steps: int):
    """Evaluate frozen models on fresh trajectories; transitions are never saved for training."""
    rows, dimensions, rewards = [], [], []
    for regime_index, regime in enumerate(("A", "B", "D", "C")):
        for seed in seeds:
            env = RecurrentHalfCheetah([(regime, steps)])
            rng = np.random.default_rng(seed + 4000 + regime_index * 10000)
            states, actions, next_states, observed_rewards = [], [], [], []
            terminated_flags, truncated_flags = [], []
            try:
                state, _ = env.reset(seed=seed)
                for _ in range(steps):
                    action = rng.uniform(-0.8, 0.8, size=6).astype(np.float32)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    states.append(state.copy())
                    actions.append(action.copy())
                    next_states.append(next_state.copy())
                    observed_rewards.append(reward)
                    terminated_flags.append(terminated)
                    truncated_flags.append(truncated)
                    state = next_state
                    if terminated or truncated:
                        break
            finally:
                env.close()
            data = {"state": np.asarray(states, dtype=np.float32),
                    "action": np.asarray(actions, dtype=np.float32),
                    "next_state": np.asarray(next_states, dtype=np.float32),
                    "reward": np.asarray(observed_rewards, dtype=np.float32),
                    "terminated": np.asarray(terminated_flags, dtype=bool),
                    "truncated": np.asarray(truncated_flags, dtype=bool)}
            for (kind, model_seed), model in models.items():
                if regime == "C" and kind == "oracle":
                    continue
                if kind == "pooled":
                    context = None
                elif kind == "oracle":
                    context = np.repeat(np.eye(3, dtype=np.float32)[regime_index:regime_index + 1],
                                        len(data["state"]), axis=0)
                else:
                    context = observed_histories(data)
                prediction = predict_state(model, data["state"], data["action"], context)
                squared = (prediction - data["next_state"]) ** 2
                rows.append({"method": kind, "seed": model_seed, "eval_seed": seed, "regime": regime,
                             "split": "heldout_online_eval", "raw_mse": float(squared.mean()),
                             "normalized_mse": normalizer.normalized_mse(
                                 prediction, data["next_state"])})
                for dimension in range(17):
                    dimensions.append({"method": kind, "seed": model_seed, "eval_seed": seed, "regime": regime,
                                       "split": "heldout_online_eval", "dimension": dimension,
                                       "raw_mse": float(squared[:, dimension].mean()),
                                       "normalized_mse": float((squared[:, dimension] /
                                           normalizer.delta_scale[dimension] ** 2).mean())})
            ridge_prediction = ridge.predict(data["state"], data["action"])
            ridge_squared = (ridge_prediction - data["next_state"]) ** 2
            rows.append({"method": "ridge", "seed": -1, "eval_seed": seed, "regime": regime,
                         "split": "heldout_online_eval", "raw_mse": float(ridge_squared.mean()),
                         "normalized_mse": normalizer.normalized_mse(
                             ridge_prediction, data["next_state"])})
            for dimension in range(17):
                dimensions.append({"method": "ridge", "seed": -1, "eval_seed": seed, "regime": regime,
                                   "split": "heldout_online_eval", "dimension": dimension,
                                   "raw_mse": float(ridge_squared[:, dimension].mean()),
                                   "normalized_mse": float((ridge_squared[:, dimension] /
                                       normalizer.delta_scale[dimension] ** 2).mean())})
            with torch.no_grad():
                predicted_reward = reward_model(torch.as_tensor(data["state"]),
                                                torch.as_tensor(data["action"])).numpy()
            rewards.append({"regime": regime, "seed": seed, "eval_seed": seed, "split": "heldout_online_eval",
                            "reward_mse": float(np.mean((predicted_reward - data["reward"]) ** 2))})
    return rows, dimensions, rewards


def rollout(models: dict, ridge: LinearRidgeDynamics, seeds: list[int]) -> list[dict]:
    rows = []
    for regime in ("A", "B", "D", "C"):
        for seed in seeds:
            env = RecurrentHalfCheetah([(regime, 8)])
            rng = np.random.default_rng(seed + 30)
            try:
                state, _ = env.reset(seed=seed)
                history = np.zeros(HISTORY_STEPS * HISTORY_WIDTH, np.float32)
                for _ in range(3):
                    action = rng.uniform(-0.8, 0.8, 6)
                    next_state, _, _, _, _ = env.step(action)
                    history = append_history(history, state, action, next_state)
                    state = next_state
                forecasts = {"ridge": (state.copy(), history.copy())}
                forecasts.update({kind: (state.copy(), history.copy()) for kind in models
                                  if not (regime == "C" and kind == "oracle")})
                for horizon, action in enumerate(rng.uniform(-0.8, 0.8, (5, 6)), start=1):
                    for kind, (forecast, old_history) in list(forecasts.items()):
                        if kind == "ridge":
                            next_forecast = ridge.predict(forecast, action)
                        else:
                            context = None if kind == "pooled" else (
                                np.eye(3, dtype=np.float32)[REGIMES.index(regime)][None]
                                if kind == "oracle" else old_history[None])
                            next_forecast = predict_state(models[kind], forecast[None],
                                                          action[None], context)[0]
                        new_history = (append_history(old_history, forecast, action, next_forecast)
                                       if kind == "history" else old_history)
                        forecasts[kind] = (next_forecast, new_history)
                    actual, _, _, _, _ = env.step(action)
                    if horizon in (1, 3, 5):
                        for kind, (prediction, _) in forecasts.items():
                            rows.append({"regime": regime, "seed": seed, "method": kind,
                                         "horizon": horizon,
                                         "raw_mse": float(np.mean((prediction - actual) ** 2)),
                                         "finite": bool(np.isfinite(prediction).all())})
            finally:
                env.close()
    return rows


def coverage(data_by: dict) -> list[dict]:
    rows = []
    for split, by_regime in data_by.items():
        for regime, data in by_regime.items():
            velocity = data["state"][:, 8]
            rows.append({"split": split, "regime": regime, "count": len(velocity),
                         "velocity_mean": float(velocity.mean()),
                         "velocity_p10": float(np.quantile(velocity, 0.1)),
                         "velocity_p90": float(np.quantile(velocity, 0.9)),
                         "fraction_velocity_gt_0_5": float((velocity > 0.5).mean()),
                         "mean_abs_action": float(np.abs(data["action"]).mean()),
                         "fraction_abs_action_gt_0_8": float((np.abs(data["action"]) > .8).mean())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/rwt/configs/stage3b/cpu_smoke.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--check-gpu", action="store_true")
    args = parser.parse_args()
    if args.resume and args.evaluate_only:
        parser.error("--resume and --evaluate-only cannot be combined")
    if args.check_gpu:
        import gymnasium, mujoco
        available = torch.cuda.is_available()
        print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                          "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__,
                          "cuda_available": available,
                          "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                          if available else []}, indent=2))
        if not available:
            raise SystemExit(2)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured_methods = tuple(config.get("methods", METHODS))
    if (not configured_methods or len(set(configured_methods)) != len(configured_methods)
            or any(method not in METHODS for method in configured_methods)
            or "pooled" not in configured_methods):
        raise ValueError("pilot methods must be distinct known methods and include pooled")
    torch.set_num_threads(int(config["cpu_threads"]))
    data_dir, output = Path(config["data_dir"]), Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    source = {"train": {r: load_split(data_dir, "train", r) for r in REGIMES},
              "validation": {r: load_split(data_dir, "validation", r) for r in REGIMES}}
    audit = dataset_audit(data_dir, source["train"], source["validation"],
                          data_dir.parent / "data_manifest.json")
    extra_dir = Path(config["extra_train_dir"]) if config.get("extra_train_dir") else None
    if extra_dir is not None:
        extra, extra_audit = load_train_addendum(
            extra_dir, set(audit["train_seeds"]) | set(audit["validation_seeds"]))
        extra_hashes = _row_hashes(pool(extra))
        if (extra_hashes & _row_hashes(pool(source["train"]))
                or extra_hashes & _row_hashes(pool(source["validation"]))):
            raise ValueError("coverage addendum duplicates original train/validation transitions")
        for regime in REGIMES:
            source["train"][regime] = {key: np.concatenate((source["train"][regime][key],
                                                            extra[regime][key]))
                                          for key in source["train"][regime]}
        audit.update(extra_audit)
    train, val = pool(source["train"]), pool(source["validation"])
    if args.resume or args.evaluate_only:
        saved_config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
        if saved_config["config"] != config or saved_config["device"] != args.device:
            raise ValueError("requested config/device differs from original run")
    else:
        (output / "resolved_config.json").write_text(json.dumps({"config": config,
            "device": args.device, "config_path": str(args.config), "torch": torch.__version__},
            indent=2), encoding="utf-8")
    hashed_files = sorted(data_dir.glob("*.npz")) + (sorted(extra_dir.glob("*.npz")) if extra_dir else [])
    data_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in hashed_files}
    if args.resume or args.evaluate_only:
        recorded_hashes = json.loads((output / "dataset_hashes.json").read_text(encoding="utf-8"))
        legacy_hashes = ({Path(path).name: digest for path, digest in data_hashes.items()}
                         if extra_dir is None else None)
        if data_hashes != recorded_hashes and (legacy_hashes is None or legacy_hashes != recorded_hashes):
            raise ValueError("dataset changed since pilot started")
        if data_hashes != recorded_hashes:
            (output / "dataset_hashes.json").write_text(json.dumps(data_hashes, indent=2), encoding="utf-8")
    else:
        (output / "dataset_hashes.json").write_text(json.dumps(data_hashes, indent=2), encoding="utf-8")
    normalizer = TrainNormalizer.fit(train["state"], train["action"], train["next_state"])
    ridge = LinearRidgeDynamics(normalizer, ridge=float(config["ridge"])).fit(
        train["state"], train["action"], train["next_state"])
    np.savez_compressed(output / "ridge.npz", weights=ridge.weights,
                        state_mean=normalizer.state_mean, state_scale=normalizer.state_scale,
                        action_mean=normalizer.action_mean, action_scale=normalizer.action_scale,
                        delta_mean=normalizer.delta_mean, delta_scale=normalizer.delta_scale)
    models, curves, runs = {}, [], []
    training_workers = int(config.get("training_workers", 1))
    if training_workers < 1 or (training_workers > 1 and args.device != "cuda"):
        raise ValueError("parallel training workers require CUDA and a positive worker count")
    jobs = []
    for kind in (*configured_methods, "reward"):
        train_context = context_for(source["train"], kind) if kind != "reward" else None
        val_context = context_for(source["validation"], kind) if kind != "reward" else None
        for seed in config["training_seeds"]:
            jobs.append((kind, int(seed), train, val, train_context, val_context,
                         config["training"], output, args.device, args.resume))
    training_start = perf_counter()
    if args.evaluate_only:
        prior_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if prior_summary["device_training"] != args.device:
            raise ValueError("evaluation device differs from saved pilot")
        selected = {kind: int(seed) for kind, seed in prior_summary["selected_seed_abd_only"].items()}
        runs = prior_summary["training_runs"]
        training_wall_seconds = float(prior_summary["training_wall_seconds"])
        hidden = int(config["training"]["hidden_dim"])
        for kind in configured_methods:
            for seed in config["training_seeds"]:
                checkpoint = torch.load(output / "checkpoints" / f"{kind}_{seed}_best.pt",
                                        map_location="cpu", weights_only=True)
                if (checkpoint["kind"] != kind or int(checkpoint["seed"]) != int(seed)
                        or checkpoint["offline_regimes"] != list(REGIMES)):
                    raise ValueError(f"checkpoint provenance mismatch for {kind}/{seed}")
                model = _model(kind, hidden)
                model.load_state_dict(checkpoint["model_state"])
                models[(kind, int(seed))] = model.eval()
        reward_seed = selected["reward"]
        reward_checkpoint = torch.load(
            output / "checkpoints" / f"reward_{reward_seed}_best.pt",
            map_location="cpu", weights_only=True)
        if (reward_checkpoint["kind"] != "reward"
                or int(reward_checkpoint["seed"]) != reward_seed
                or reward_checkpoint["offline_regimes"] != list(REGIMES)):
            raise ValueError("reward checkpoint provenance mismatch")
        reward_model = RewardPredictor(hidden)
        reward_model.load_state_dict(reward_checkpoint["model_state"])
        models[("reward", reward_seed)] = reward_model.eval()
    elif training_workers == 1:
        trained = [_train(*job) for job in jobs]
    else:
        # Spawn avoids inheriting a CUDA context and gives each seed isolated RNG state.
        with ProcessPoolExecutor(max_workers=training_workers,
                                 mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(_train, *job) for job in jobs]
            trained = [future.result() for future in futures]
    if not args.evaluate_only:
        training_wall_seconds = perf_counter() - training_start
        for job, (model, curve, row) in zip(jobs, trained):
            kind, seed = job[:2]
            models[(kind, seed)] = model
            curves.extend(curve)
            runs.append(row)
        selected = {kind: min((row for row in runs if row["kind"] == kind),
                              key=lambda row: row["validation_loss"])["seed"]
                    for kind in (*configured_methods, "reward")}
    chosen = {kind: models[(kind, selected[kind])] for kind in (*configured_methods, "reward")}
    if not args.evaluate_only:
        write_csv(output / "training_curves.csv", curves)
        write_csv(output / "training_runs.csv", runs)
    one, dimensions, rewards = validation_metrics(
        {(kind, seed): model for (kind, seed), model in models.items() if kind != "reward"},
        chosen["reward"], source["validation"], normalizer)
    for regime in REGIMES:
        data = source["validation"][regime]
        prediction = ridge.predict(data["state"], data["action"])
        squared = (prediction - data["next_state"]) ** 2
        one.append({"method": "ridge", "seed": -1, "eval_seed": -1, "regime": regime,
                    "split": "validation_abd",
                    "raw_mse": float(squared.mean()),
                    "normalized_mse": normalizer.normalized_mse(prediction, data["next_state"])})
        for dimension in range(17):
            dimensions.append({"method": "ridge", "seed": -1, "eval_seed": -1, "regime": regime,
                               "split": "validation_abd",
                               "dimension": dimension, "raw_mse": float(squared[:, dimension].mean()),
                               "normalized_mse": float((squared[:, dimension] /
                                                        normalizer.delta_scale[dimension] ** 2).mean())})
    heldout_one, heldout_dimensions, heldout_rewards = heldout_metrics(
        {(kind, seed): model for (kind, seed), model in models.items() if kind != "reward"},
        ridge, chosen["reward"], normalizer, [int(x) for x in config["heldout_seeds"]],
        int(config["heldout_steps"]))
    one.extend(heldout_one)
    dimensions.extend(heldout_dimensions)
    rewards.extend(heldout_rewards)
    write_csv(output / "one_step.csv", one)
    write_csv(output / "per_dimension.csv", dimensions)
    write_csv(output / "reward_validation.csv", rewards)
    write_csv(output / "coverage.csv", coverage(source))
    # Every C evaluation below follows A/B/D-only checkpoint selection above.
    roll = rollout({kind: chosen[kind] for kind in configured_methods}, ridge,
                   [int(x) for x in config["rollout_seeds"]])
    write_csv(output / "rollout.csv", roll)
    world = SeparatedWorldModel(chosen["pooled"].to(args.device),
                                chosen["reward"].to(args.device)).eval()
    support = TrainSupport(train["state"], train["action"], val["state"], val["action"])
    plans, steps = mpc_replay(world, support, [int(x) for x in config["mpc_seeds"]], horizons=(3,))
    write_csv(output / "mpc_plans.csv", plans)
    write_csv(output / "mpc_plan_steps.csv", steps)
    summary = {"dataset": audit, "device_training": args.device,
               "training_workers": training_workers, "training_wall_seconds": training_wall_seconds,
               "selected_seed_abd_only": selected, "training_runs": runs,
               "validation_support_threshold": support.threshold,
               "rollout_nonfinite": sum(not row["finite"] for row in roll),
               "mpc_invalid_candidates": sum(row["invalid_candidates"] for row in plans)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
