"""Stage 3A HalfCheetah CPU smoke pipeline; also accepts a larger GPU config.

Run from repository root with PYTHONPATH=src (or an editable installation).
Offline files contain transitions/rewards only. Regime names stay in collection
and evaluator metadata; the learner, residual adapter and planner never see them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
import yaml

from rwt.stage3a import (BatchedMiniCEM, CEMConfig, DEFAULT_ORDER, LinearResidualAdapter,
                         MiniCEM, MiniDynamics, PhysicsRegime,
                         RecurrentHalfCheetah, load_checkpoint, save_checkpoint)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _regimes(config: dict) -> dict[str, PhysicsRegime]:
    regimes = {name: PhysicsRegime(float(p["friction"]), float(p["actuator_gain"]))
               for name, p in config["regimes"].items()}
    if set(regimes) != set(DEFAULT_ORDER):
        raise ValueError("Stage 3A requires exactly regimes A/B/D/C")
    return regimes


def _collect_one(regime: str, count: int, episode_steps: int, seed: int,
                 regimes: dict[str, PhysicsRegime], path: Path) -> dict:
    if count <= 0 or episode_steps <= 0:
        raise ValueError("collection counts must be positive")
    rng = np.random.default_rng(seed)
    episodes = 0
    start = perf_counter()
    env = RecurrentHalfCheetah([(regime, episode_steps)], regimes)
    rows = {"state": np.empty((count, env.observation_space.shape[0]), dtype=np.float32),
            "action": np.empty((count, env.action_space.shape[0]), dtype=np.float32),
            "next_state": np.empty((count, env.observation_space.shape[0]), dtype=np.float32),
            "reward": np.empty(count, dtype=np.float32),
            "terminated": np.empty(count, dtype=np.bool_),
            "truncated": np.empty(count, dtype=np.bool_)}
    index = 0
    try:
        while index < count:
            state, _ = env.reset(seed=seed + episodes)
            previous_action = np.zeros(env.action_space.shape, dtype=np.float64)
            episodes += 1
            for _ in range(episode_steps):
                # Mixed smooth and exploratory controls; no regime data enters
                # the action generator. Bounds are those of the real environment.
                action = np.clip(0.55 * previous_action + 0.45 * rng.normal(0, 0.8, env.action_space.shape),
                                 env.action_space.low, env.action_space.high)
                next_state, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(next_state).all() and np.isfinite(reward)):
                    raise FloatingPointError("nonfinite MuJoCo collection transition")
                rows["state"][index] = state
                rows["action"][index] = action
                rows["next_state"][index] = next_state
                rows["reward"][index] = reward
                rows["terminated"][index] = terminated
                rows["truncated"][index] = truncated
                index += 1
                state, previous_action = next_state, action
                if index >= count or terminated or truncated:
                    break
    finally:
        env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **rows)
    elapsed = perf_counter() - start
    return {"regime": regime, "count": index, "seed": seed, "episodes": episodes,
            "seconds": elapsed, "transitions_per_second": count / elapsed,
            "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _load_pooled(paths: list[Path]) -> dict[str, np.ndarray]:
    if not paths or any(path.stem.split("_")[-1] not in {"A", "B", "D"} for path in paths):
        raise ValueError("training and validation accept A/B/D files only")
    arrays = []
    for path in paths:
        with np.load(path) as data:
            arrays.append({key: data[key].copy() for key in ("state", "action", "next_state", "reward")})
    return {key: np.concatenate([item[key] for item in arrays]) for key in arrays[0]}


def _tensors(data: dict[str, np.ndarray], device: str):
    return tuple(torch.as_tensor(data[name], dtype=torch.float32, device=device)
                 for name in ("state", "action", "next_state", "reward"))


def _train(model: MiniDynamics, train: dict, validation: dict, config: dict,
           device: str, checkpoint: Path, *, training_state: Path | None = None,
           resume: bool = False) -> tuple[list[dict], float]:
    torch.manual_seed(int(config["seed"]))
    model.to(device)
    if not resume:
        model.fit_normalization(train["state"], train["action"], train["next_state"], train["reward"])
    train_t, val_t = _tensors(train, device), _tensors(validation, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    batch_size = int(config["batch_size"])
    on_cuda = device.startswith("cuda")
    history: list[dict] = []
    best, start_epoch = np.inf, 1
    if resume:
        if training_state is None or not training_state.exists() or not checkpoint.exists():
            raise FileNotFoundError("resume requires training_state.pt and best_dynamics.pt")
        payload = torch.load(training_state, map_location=device, weights_only=True)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        history = payload["history"]
        best = float(payload["best_validation_loss"])
        start_epoch = int(payload["epoch"]) + 1
        torch.set_rng_state(payload["cpu_rng_state"].cpu())
        if on_cuda:
            if "cuda_rng_state" not in payload:
                raise ValueError("CUDA resume state is missing CUDA RNG")
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
        if start_epoch > int(config["epochs"]):
            return history, 0.0
    start = perf_counter()
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train_t[0]), device=device)
        weighted_train_loss = torch.zeros((), device=device)
        finite_epoch = torch.ones((), device=device, dtype=torch.bool)
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(*(array[indices] for array in train_t))
            if on_cuda:
                finite_epoch.logical_and_(torch.isfinite(loss.detach()))
            elif not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            if not on_cuda and not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite model gradient")
            optimizer.step()
            weighted_train_loss += loss.detach() * len(indices)
        if on_cuda and not bool(finite_epoch.item()):
            raise FloatingPointError("nonfinite CUDA training loss")
        if on_cuda and not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all().item()):
            raise FloatingPointError("nonfinite CUDA model parameters")
        model.eval()
        with torch.no_grad():
            val_loss = float(model.loss(*val_t))
        if not np.isfinite(val_loss):
            raise FloatingPointError("nonfinite validation loss")
        history.append({"epoch": epoch, "train_loss": float(weighted_train_loss.item()) / len(train_t[0]),
                        "validation_loss": val_loss})
        if val_loss < best:
            best = val_loss
            save_checkpoint(model, checkpoint, {"best_epoch": epoch, "validation_loss": val_loss,
                                                "offline_regimes": ["A", "B", "D"]})
        if training_state is not None:
            training_state.parent.mkdir(parents=True, exist_ok=True)
            payload = {"epoch": epoch, "model_state": model.state_dict(),
                       "optimizer_state": optimizer.state_dict(), "history": history,
                       "best_validation_loss": best, "cpu_rng_state": torch.get_rng_state()}
            if on_cuda:
                payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
            temporary = training_state.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(training_state)
    return history, perf_counter() - start


def _evaluate(model: MiniDynamics, config: dict, regimes: dict[str, PhysicsRegime],
              device: str) -> tuple[list[dict], list[dict]]:
    evaluation = config["evaluation"]
    dwell = [int(x) for x in evaluation["dwell_times"]]
    if len(dwell) != len(DEFAULT_ORDER):
        raise ValueError("dwell_times must match A/B/D/C/A/C")
    schedule = list(zip(DEFAULT_ORDER, dwell))
    cem_config = CEMConfig(**config["mpc"])
    episode_rows, step_rows = [], []
    for seed in evaluation["seeds"]:
        env = RecurrentHalfCheetah(schedule, regimes)
        adapter = LinearResidualAdapter.from_model(model, ridge=float(evaluation["adapter_ridge"]))
        planner = MiniCEM(model, env.action_space.low, env.action_space.high,
                          cem_config, seed=int(seed), adapter=adapter, diagnostics=True)
        episode_return = 0.0
        runtimes = []
        try:
            state, _ = env.reset(seed=int(seed))
            for step_index in range(sum(dwell)):
                action = planner.act(state)
                runtimes.append(planner.last_runtime_seconds)
                if not env.action_space.contains(action.astype(env.action_space.dtype)):
                    raise AssertionError("MPC action outside environment bounds")
                with torch.no_grad():
                    state_t = torch.as_tensor(state[None], dtype=torch.float32, device=device)
                    action_t = torch.as_tensor(action[None], dtype=torch.float32, device=device)
                    next_mean, reward_mean = model(state_t, action_t)
                    raw_prediction = next_mean[0].cpu().numpy()
                    adapted_prediction = raw_prediction + adapter.predict(state, action)
                next_state, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(next_state).all() and np.isfinite(reward)):
                    raise FloatingPointError("nonfinite MuJoCo evaluation transition")
                # The evaluator alone reads the label. The adapter updates from
                # this observed transition only, after pre-update metrics.
                transition_regime = env.last_transition_regime
                step_rows.append({"seed": seed, "step": step_index, "regime": transition_regime,
                                  "pre_update_mse": float(np.mean((adapted_prediction - next_state) ** 2)),
                                  "base_mse": float(np.mean((raw_prediction - next_state) ** 2)),
                                  "reward_error": float(reward_mean.item() - reward),
                                  "reward": float(reward), "mpc_seconds": runtimes[-1],
                                  "planned_horizon_return": planner.last_predicted_return,
                                  "plan_ood_fraction": planner.last_plan_ood_fraction,
                                  "invalid_candidates": planner.last_invalid_candidates})
                adapter.update(state, action, next_state - raw_prediction)
                episode_return += float(reward)
                state = next_state
                if terminated or truncated:
                    break
            episode_rows.append({"seed": seed, "steps": len(runtimes),
                                 "return": episode_return,
                                 "mpc_mean_seconds": float(np.mean(runtimes)),
                                 "mpc_p95_seconds": float(np.quantile(runtimes, 0.95)),
                                 "adapter_updates": adapter.updates,
                                 "adapter_parameters": adapter.parameter_count})
        finally:
            env.close()
    return episode_rows, step_rows


def _evaluate_batched(model: MiniDynamics, config: dict, regimes: dict[str, PhysicsRegime],
                      device: str) -> tuple[list[dict], list[dict]]:
    """Step independent MuJoCo episodes on CPU; batch all MPC rollouts on GPU."""
    evaluation = config["evaluation"]
    dwell = [int(x) for x in evaluation["dwell_times"]]
    if len(dwell) != len(DEFAULT_ORDER):
        raise ValueError("dwell_times must match A/B/D/C/A/C")
    schedule = list(zip(DEFAULT_ORDER, dwell))
    seeds = [int(seed) for seed in evaluation["seeds"]]
    envs = [RecurrentHalfCheetah(schedule, regimes) for _ in seeds]
    adapters = [LinearResidualAdapter.from_model(model, ridge=float(evaluation["adapter_ridge"]))
                for _ in seeds]
    planner = BatchedMiniCEM(model, envs[0].action_space.low, envs[0].action_space.high,
                             CEMConfig(**config["mpc"]), seeds)
    episode_returns = np.zeros(len(seeds), dtype=np.float64)
    runtimes = [[] for _ in seeds]
    steps = np.zeros(len(seeds), dtype=np.int64)
    active = np.ones(len(seeds), dtype=np.bool_)
    episode_rows, step_rows = [], []
    try:
        states = [env.reset(seed=seed)[0] for env, seed in zip(envs, seeds)]
        for step_index in range(sum(dwell)):
            indices = np.flatnonzero(active).tolist()
            if not indices:
                break
            batch_states = np.stack([states[i] for i in indices])
            actions = planner.act(batch_states, [adapters[i] for i in indices], indices)
            per_action_seconds = planner.last_runtime_seconds / len(indices)
            with torch.no_grad():
                state_t = torch.as_tensor(batch_states, dtype=torch.float32, device=device)
                action_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
                next_mean, reward_mean = model.predict(state_t, action_t)
                raw_predictions = next_mean.cpu().numpy()
                predicted_rewards = reward_mean.cpu().numpy()
            for j, i in enumerate(indices):
                action = actions[j]
                if not envs[i].action_space.contains(action.astype(envs[i].action_space.dtype)):
                    raise AssertionError("batched MPC action outside environment bounds")
                state = states[i]
                raw_prediction = raw_predictions[j]
                adapted_prediction = raw_prediction + adapters[i].predict(state, action)
                next_state, reward, terminated, truncated, _ = envs[i].step(action)
                if not (np.isfinite(next_state).all() and np.isfinite(reward)):
                    raise FloatingPointError("nonfinite MuJoCo evaluation transition")
                step_rows.append({"seed": seeds[i], "step": step_index,
                                  "regime": envs[i].last_transition_regime,
                                  "pre_update_mse": float(np.mean((adapted_prediction - next_state) ** 2)),
                                  "base_mse": float(np.mean((raw_prediction - next_state) ** 2)),
                                  "reward_error": float(predicted_rewards[j] - reward),
                                  "reward": float(reward), "mpc_seconds": per_action_seconds,
                                  "planned_horizon_return": float(planner.last_predicted_return[i]),
                                  "plan_ood_fraction": float(planner.last_plan_ood_fraction[i]),
                                  "invalid_candidates": int(planner.last_invalid_candidates[i])})
                adapters[i].update(state, action, next_state - raw_prediction)
                episode_returns[i] += float(reward)
                runtimes[i].append(per_action_seconds)
                steps[i] += 1
                states[i] = next_state
                if terminated or truncated:
                    active[i] = False
        for i, seed in enumerate(seeds):
            episode_rows.append({"seed": seed, "steps": int(steps[i]),
                                 "return": float(episode_returns[i]),
                                 "mpc_mean_seconds": float(np.mean(runtimes[i])),
                                 "mpc_p95_seconds": float(np.quantile(runtimes[i], 0.95)),
                                 "adapter_updates": adapters[i].updates,
                                 "adapter_parameters": adapters[i].parameter_count})
    finally:
        for env in envs:
            env.close()
    return episode_rows, step_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/rwt/configs/stage3a/cpu_smoke.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="resume from output/checkpoints/training_state.pt")
    parser.add_argument("--check-gpu", action="store_true", help="inspect CUDA environment and exit without training")
    args = parser.parse_args()
    if args.check_gpu:
        import mujoco
        available = torch.cuda.is_available()
        gpu_info = {"python": platform.python_version(), "torch": torch.__version__,
                    "gymnasium": gym.__version__, "mujoco": mujoco.__version__,
                    "cuda_available": available, "device_count": torch.cuda.device_count() if available else 0,
                    "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                    if available else []}
        if available:
            sample = torch.ones((2, 2), device="cuda")
            gpu_info["cuda_matmul_finite"] = bool(torch.isfinite(sample @ sample).all().item())
        print(json.dumps(gpu_info, indent=2))
        if not available:
            raise SystemExit(2)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if tuple(config["offline"]["regimes"]) != ("A", "B", "D"):
        raise ValueError("offline training/validation must use A, B, D only")
    if args.device == "cpu":
        torch.set_num_threads(int(config.get("cpu_threads", 2)))
    output = args.output or Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    prior_metrics_path = output / "metrics.json"
    prior_metrics = (json.loads(prior_metrics_path.read_text(encoding="utf-8"))
                     if args.resume and prior_metrics_path.exists() else {})
    regimes = _regimes(config)
    config_path = output / "resolved_config.json"
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError("resume requires existing resolved_config.json")
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old["config"] != config or old["device"] != args.device:
            raise ValueError("resume config/device differs from original run")
    else:
        with config_path.open("w", encoding="utf-8") as stream:
            json.dump({"config": config, "device": args.device, "config_path": str(args.config),
                       "python": platform.python_version(), "gymnasium": gym.__version__,
                       "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}, stream, indent=2)
    env = RecurrentHalfCheetah([("A", 1)], regimes)
    observation, _ = env.reset(seed=int(config["seed"]))
    state_dim, action_dim = observation.shape[0], env.action_space.shape[0]
    env.close()
    if (state_dim, action_dim) != (17, 6):
        raise RuntimeError(f"unexpected HalfCheetah-v5 dimensions: {(state_dim, action_dim)}")

    offline = config["offline"]
    workers = int(offline.get("collection_workers", 1))
    if workers < 1:
        raise ValueError("collection_workers must be positive")
    if args.resume:
        manifest_path = output / "data_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["offline_regimes"] != ["A", "B", "D"] or manifest["held_out_regime"] != "C":
            raise ValueError("resume dataset manifest violates held-out C split")
        collection = manifest["splits"]
        for item in collection:
            if "sha256" in item:
                filename = Path(item["file"]).name
                if filename not in {f"{split}_{regime}.npz" for split in ("train", "validation")
                                    for regime in ("A", "B", "D")}:
                    raise ValueError("resume manifest contains an unexpected dataset path")
                path = output / "data" / filename
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError(f"resume dataset changed: {path}")
        collection_wall_seconds = prior_metrics.get("collection_wall_seconds")
    else:
        jobs = []
        for split, count in (("train", int(offline["train_count_per_regime"])),
                             ("validation", int(offline["validation_count_per_regime"]))):
            for i, regime in enumerate(offline["regimes"]):
                seed = int(config["seed"]) + (0 if split == "train" else 100000) + 1000 * i
                jobs.append((regime, count, int(offline["episode_steps"]), seed,
                             regimes, output / "data" / f"{split}_{regime}.npz"))
        collection_start = perf_counter()
        if workers == 1:
            collection = [_collect_one(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                     mp_context=get_context("spawn")) as executor:
                futures = [executor.submit(_collect_one, *job) for job in jobs]
                collection = [future.result() for future in futures]
        collection_wall_seconds = perf_counter() - collection_start
        manifest = {"offline_regimes": list(offline["regimes"]), "held_out_regime": "C",
                    "splits": collection, "dataset_fields": ["state", "action", "next_state", "reward", "terminated", "truncated"]}
        with (output / "data_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
    train = _load_pooled([output / "data" / f"train_{r}.npz" for r in offline["regimes"]])
    validation = _load_pooled([output / "data" / f"validation_{r}.npz" for r in offline["regimes"]])
    torch.manual_seed(int(config["seed"]))
    model = MiniDynamics(state_dim, action_dim, int(config["model"]["hidden_dim"]))
    checkpoint = output / "checkpoints" / "best_dynamics.pt"
    training_state = output / "checkpoints" / "training_state.pt"
    history, training_seconds = _train(model, train, validation, {**config["training"], "seed": config["seed"]},
                                       args.device, checkpoint, training_state=training_state, resume=args.resume)
    _write_csv(output / "training_curve.csv", history)
    model, checkpoint_info = load_checkpoint(checkpoint, args.device)
    with torch.no_grad():
        validation_t = _tensors(validation, args.device)
        validation_mean, validation_reward = model.predict(validation_t[0], validation_t[1])
        offline_validation_mse = float((validation_mean - validation_t[2]).square().mean())
        offline_validation_reward_mse = float((validation_reward - validation_t[3]).square().mean())
    batched_evaluation = bool(config["evaluation"].get("batch_seeds", False))
    episode_rows, step_rows = (_evaluate_batched if batched_evaluation else _evaluate)(
        model, config, regimes, args.device)
    _write_csv(output / "episodes.csv", episode_rows)
    _write_csv(output / "evaluation_steps.csv", step_rows)
    metrics = {"state_dim": state_dim, "action_dim": action_dim,
               "collected_transitions": int(sum(row["count"] for row in collection)),
               "collection_workers": workers,
               "collection_reused": args.resume,
               "collection_wall_seconds": collection_wall_seconds,
               "simulator_transitions_per_second": (float(sum(row["count"] for row in collection) /
                                                          collection_wall_seconds)
                                                    if collection_wall_seconds is not None else None),
               "training_seconds": float(prior_metrics.get("training_seconds", 0.0) + training_seconds),
               "training_seconds_this_invocation": training_seconds,
               "best_epoch": checkpoint_info["best_epoch"],
               "validation_loss": checkpoint_info["validation_loss"],
               "offline_validation_mse": offline_validation_mse,
               "offline_validation_reward_mse": offline_validation_reward_mse,
               "evaluation_seeds": len(episode_rows),
               "evaluation_batched_seeds": batched_evaluation,
               "mpc_total_seconds": float(sum(row["mpc_seconds"] for row in step_rows)),
               "short_episode_return_mean": float(np.mean([row["return"] for row in episode_rows])),
               "pre_update_mse": float(np.mean([row["pre_update_mse"] for row in step_rows])),
               "base_mse": float(np.mean([row["base_mse"] for row in step_rows])),
               "mpc_mean_seconds": float(np.mean([row["mpc_seconds"] for row in step_rows])),
               "mpc_p95_seconds": float(np.quantile([row["mpc_seconds"] for row in step_rows], 0.95)),
               "mpc_ood_steps": int(sum(row["plan_ood_fraction"] > 0 for row in step_rows)),
               "mpc_invalid_candidates": int(sum(row["invalid_candidates"] for row in step_rows)),
               "per_regime_pre_update_mse": {r: float(np.mean([row["pre_update_mse"] for row in step_rows
                                                               if row["regime"] == r])) for r in DEFAULT_ORDER}}
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
