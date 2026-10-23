"""M2 CPU training and diagnostics. M2 checkpoints freeze before M3."""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from q1.datasets import Trajectory, rollout
from q1.model import ContextDynamicsModel, ContextFreeDynamicsModel, TransitionNormalizer
from q1.simulator import PROGRAMS, DynamicsParams, apply_program, sample_params


@dataclass
class Task:
    support: np.ndarray
    query: np.ndarray
    query_states: np.ndarray
    query_actions: np.ndarray
    params: DynamicsParams
    env_id: str


def _transitions(trajectory: Trajectory) -> np.ndarray:
    return np.concatenate((trajectory.states[:-1], trajectory.actions, trajectory.delta_v), axis=1).astype(np.float32)


def _make_tasks(manifest: dict, split: str, count: int, *, seed: int, support_n: int,
                query_n: int, dt: float, sigma: float, parameter_ranges: dict,
                programs: list[tuple[int, ...]]) -> list[Task]:
    tasks = []
    for env_idx in range(count):
        env_id = manifest["environment_ids"][split][env_idx]
        params = sample_params(np.random.default_rng(manifest["environment_seeds"][split][env_idx]),
                               manifest["regions"][split], parameter_ranges)
        for p_idx, program in enumerate(programs):
            target_params = apply_program(params, program)
            base_seed = seed + env_idx * 100003 + p_idx * 7919
            support_traj = rollout(target_params, n=support_n, seed=base_seed, dt=dt, sigma=sigma)
            query_traj = rollout(target_params, n=query_n, seed=base_seed + 17, dt=dt, sigma=sigma)
            tasks.append(Task(_transitions(support_traj), _transitions(query_traj),
                              query_traj.states, query_traj.actions, target_params, env_id))
    return tasks


def _torch_support(task: Task, idx: np.ndarray, normalizer: TransitionNormalizer, device: torch.device) -> torch.Tensor:
    data = torch.as_tensor(task.support[idx], dtype=torch.float32, device=device)
    return torch.cat((normalizer.normalize_x(data[:, :6]), normalizer.normalize_dv(data[:, 6:])), dim=-1)[None]


def _fit_models(train_tasks: list[Task], validation_tasks: list[Task],
                normalizer: TransitionNormalizer, config: dict,
                seed: int, device: torch.device) -> tuple[ContextDynamicsModel, ContextFreeDynamicsModel, list[dict]]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed + 89)
    context_dim = int(config["model"]["context_dim"])
    hidden = int(config["model"]["hidden_dim"])
    depth = int(config["model"].get("depth", 2))
    dropout = float(config["model"].get("dropout", 0.0))
    model = ContextDynamicsModel(context_dim, hidden, depth=depth, dropout=dropout).to(device)
    baseline = ContextFreeDynamicsModel(context_dim, hidden, depth=depth, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=float(config["training"]["learning_rate"]))
    epochs = int(config["training"]["dynamics_epochs"])
    batch_size = int(config["training"]["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train_tasks) / batch_size))
    history = []
    for epoch in range(epochs):
        model.train()
        baseline.train()
        order = np_rng.permutation(len(train_tasks))
        losses, baseline_losses, train_nlls, baseline_train_nlls = [], [], [], []
        for start in range(0, len(order), batch_size):
            task_ids = order[start:start + batch_size]
            supports, xs, ys, base_xs, base_ys = [], [], [], [], []
            for task_id in task_ids:
                task = train_tasks[int(task_id)]
                s_idx = np_rng.choice(len(task.support), size=min(64, len(task.support)), replace=False)
                q_idx = int(np_rng.integers(len(task.query)))
                supports.append(_torch_support(task, s_idx, normalizer, device)[0])
                q = torch.as_tensor(task.query[q_idx], dtype=torch.float32, device=device)
                xs.append(normalizer.normalize_x(q[:6]))
                ys.append(normalizer.normalize_dv(q[6:]))
            support_batch = torch.stack(supports)
            x_batch, y_batch = torch.stack(xs), torch.stack(ys)
            pred = normalizer.denormalize_dv(model(support_batch, x_batch))
            target = normalizer.denormalize_dv(y_batch)
            loss = torch.mean(0.5 * (math.log(2.0 * math.pi * float(config["model"]["predictive_variance"]))
                                     + (pred - target) ** 2 / float(config["model"]["predictive_variance"])))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(torch.mean((pred - target) ** 2).detach().cpu()))
            train_nlls.append(float(loss.detach().cpu()))
            base_pred = normalizer.denormalize_dv(baseline.predict(x_batch))
            base_loss = torch.mean(0.5 * (math.log(2.0 * math.pi * float(config["model"]["predictive_variance"]))
                                          + (base_pred - target) ** 2 / float(config["model"]["predictive_variance"])))
            baseline_optimizer.zero_grad(set_to_none=True)
            base_loss.backward()
            baseline_optimizer.step()
            baseline_losses.append(float(torch.mean((base_pred - target) ** 2).detach().cpu()))
            baseline_train_nlls.append(float(base_loss.detach().cpu()))
        model.eval()
        baseline.eval()
        validation_losses, validation_baseline_losses = [], []
        validation_nlls, validation_baseline_nlls = [], []
        with torch.no_grad():
            for task in validation_tasks[:min(32, len(validation_tasks))]:
                support_idx = np.arange(min(64, len(task.support)))
                support = _torch_support(task, support_idx, normalizer, device)
                query_idx = (epoch * 17) % len(task.query)
                query = torch.as_tensor(task.query[query_idx], dtype=torch.float32, device=device)
                x = normalizer.normalize_x(query[:6])
                y = normalizer.normalize_dv(query[6:])
                context = model.encode(support)
                pred = normalizer.denormalize_dv(model.predict(x[None], context)[0])
                base_pred = normalizer.denormalize_dv(baseline.predict(x[None])[0])
                truth = normalizer.denormalize_dv(y)
                residual, base_residual = pred - truth, base_pred - truth
                variance = float(config["model"]["predictive_variance"])
                validation_losses.append(float(torch.mean(residual ** 2).cpu()))
                validation_baseline_losses.append(float(torch.mean(base_residual ** 2).cpu()))
                validation_nlls.append(float(torch.mean(0.5 * (math.log(2.0 * math.pi * variance)
                                                                + residual ** 2 / variance)).cpu()))
                validation_baseline_nlls.append(float(torch.mean(0.5 * (math.log(2.0 * math.pi * variance)
                                                                        + base_residual ** 2 / variance)).cpu()))
        history.append({"epoch": epoch + 1, "train_mse_delta_v": float(np.mean(losses)),
                        "context_free_train_mse_delta_v": float(np.mean(baseline_losses)),
                        "train_nll_per_delta_v_dimension": float(np.mean(train_nlls)),
                        "context_free_train_nll_per_delta_v_dimension": float(np.mean(baseline_train_nlls)),
                        "validation_mse_delta_v": float(np.mean(validation_losses)),
                        "context_free_validation_mse_delta_v": float(np.mean(validation_baseline_losses)),
                        "validation_nll_per_delta_v_dimension": float(np.mean(validation_nlls)),
                        "context_free_validation_nll_per_delta_v_dimension": float(np.mean(validation_baseline_nlls))})
    return model, baseline, history


@torch.no_grad()
def _evaluate(model: ContextDynamicsModel, baseline: ContextFreeDynamicsModel,
              tasks: list[Task], normalizer: TransitionNormalizer, device: torch.device,
              predictive_variance: float, dt: float = 0.05, horizons=(5, 10, 20)) -> dict:
    context_sq, base_sq, context_n = 0.0, 0.0, 0
    context_nll_sum = base_nll_sum = 0.0
    support_budget_squares = {24: 0.0, 64: 0.0, 256: 0.0}
    support_budget_count = {24: 0, 64: 0, 256: 0}
    swap_errors, response_diffs = [], []
    rollout_errors = {str(h): [] for h in horizons}
    prepared = []
    for task in tasks:
        idx = np.arange(min(64, len(task.support)))
        support = _torch_support(task, idx, normalizer, device)
        context = model.encode(support)
        x = torch.as_tensor(task.query[:, :6], dtype=torch.float32, device=device)
        y = torch.as_tensor(task.query[:, 6:], dtype=torch.float32, device=device)
        xn = normalizer.normalize_x(x)
        yn = normalizer.normalize_dv(y)
        pred = normalizer.denormalize_dv(model.predict(xn, context))
        base_pred = normalizer.denormalize_dv(baseline.predict(xn))
        truth = normalizer.denormalize_dv(yn)
        context_residual = pred - truth
        base_residual = base_pred - truth
        context_sq += float(torch.sum(context_residual ** 2).cpu())
        base_sq += float(torch.sum(base_residual ** 2).cpu())
        log_constant = math.log(2.0 * math.pi * predictive_variance)
        context_nll_sum += float(torch.sum(0.5 * (log_constant + context_residual ** 2 / predictive_variance)).cpu())
        base_nll_sum += float(torch.sum(0.5 * (log_constant + base_residual ** 2 / predictive_variance)).cpu())
        context_n += truth.numel()
        for budget in support_budget_squares:
            if len(task.support) < budget:
                continue
            support_budget = _torch_support(task, np.arange(budget), normalizer, device)
            context_budget = model.encode(support_budget)
            budget_pred = normalizer.denormalize_dv(model.predict(xn, context_budget))
            support_budget_squares[budget] += float(torch.sum((budget_pred - truth) ** 2).cpu())
            support_budget_count[budget] += truth.numel()
        prepared.append((task, support, context))
        initial = task.query_states[0].copy()
        for horizon in horizons:
            state = initial.copy()
            pred_states = [state.copy()]
            for k in range(min(horizon, len(task.query_actions))):
                xrow = np.concatenate((state, task.query_actions[k])).astype(np.float32)
                xrow_t = normalizer.normalize_x(torch.as_tensor(xrow[None], device=device))
                dv = normalizer.denormalize_dv(model.predict(xrow_t, context))[0].cpu().numpy()
                v_next = state[2:] + dv
                state = np.concatenate((state[:2] + dt * v_next, v_next))
                pred_states.append(state.copy())
            truth_states = task.query_states[:len(pred_states)]
            rollout_errors[str(horizon)].append(float(np.sqrt(np.mean((np.asarray(pred_states) - truth_states) ** 2))))
    # Context swapping between distinct target laws, on fixed held-out queries.
    by_environment = {}
    for item in prepared:
        by_environment.setdefault(item[0].env_id, item)
    environment_items = list(by_environment.values())
    for i in range(0, len(environment_items) - 1, 2):
        task_a, support_a, context_a = environment_items[i]
        task_b, support_b, context_b = environment_items[i + 1]
        xa = normalizer.normalize_x(torch.as_tensor(task_a.query[:, :6], dtype=torch.float32, device=device))
        ya = torch.as_tensor(task_a.query[:, 6:], dtype=torch.float32, device=device)
        pred_right = normalizer.denormalize_dv(model.predict(xa, context_a))
        pred_swapped = normalizer.denormalize_dv(model.predict(xa, context_b))
        swap_errors.append(float(torch.mean((pred_swapped - normalizer.denormalize_dv(normalizer.normalize_dv(ya))) ** 2).cpu()))
        response_diffs.append(float(torch.sqrt(torch.mean((pred_right - pred_swapped) ** 2)).cpu()))
    return {
        "context_mse": context_sq / max(1, context_n),
        "context_free_mse": base_sq / max(1, context_n),
        "context_nll_per_delta_v_dimension": context_nll_sum / max(1, context_n),
        "context_free_nll_per_delta_v_dimension": base_nll_sum / max(1, context_n),
        "support_budget_mse": {str(k): (support_budget_squares[k] / support_budget_count[k]
                                         if support_budget_count[k] else None)
                                for k in support_budget_squares},
        "relative_improvement_over_context_free": (base_sq - context_sq) / max(base_sq, 1e-12),
        "context_swap_mse": float(np.mean(swap_errors)) if swap_errors else None,
        "context_swap_response_rmse": float(np.mean(response_diffs)) if response_diffs else None,
        "open_loop_state_rmse": {k: float(np.mean(v)) for k, v in rollout_errors.items()},
    }


def run_m2(config: dict, output_dir: str | Path, *, seed: int | None = None,
           device_name: str = "cpu") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"] if seed is None else seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((output.parent / "split_manifest.json").read_text(encoding="utf-8")) if (output.parent / "split_manifest.json").exists() else None
    if manifest is None:
        from q1.datasets import make_split_manifest
        manifest = make_split_manifest(seed, split_cfg["counts"])
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train_programs = [()] + [(i,) for i in range(4)] + [tuple(p) for p in split_cfg["composition_training_pairs"]]
    support_n, query_n = (64, 64) if config["run_kind"] == "SMOKE" else (256, 96)
    common = dict(seed=seed, support_n=support_n, query_n=query_n,
                  dt=float(config["dt"]), sigma=float(config["noise_sigma"]),
                  parameter_ranges=split_cfg["parameter_ranges"], programs=train_programs)
    train_tasks = _make_tasks(manifest, "train", int(config["oracle"]["train_envs"]), **common)
    val_tasks = _make_tasks(manifest, "validation", int(config["oracle"]["validation_envs"]), **common)
    normalizer = TransitionNormalizer.fit([task.support for task in train_tasks] + [task.query for task in train_tasks])
    model, baseline, history = _fit_models(train_tasks, val_tasks, normalizer, config, seed, device)
    model.eval()
    baseline.eval()
    metrics = _evaluate(model, baseline, val_tasks, normalizer, device,
                        predictive_variance=float(config["model"]["predictive_variance"]),
                        dt=float(config["dt"]))
    minimum_improvement = float(config["gates"]["m2_min_relative_improvement"])
    minimum_swap = float(config["gates"]["m2_min_context_swap_response_rmse"])
    gate_pass = (metrics["relative_improvement_over_context_free"] >= minimum_improvement
                 and (metrics["context_swap_response_rmse"] or 0.0) >= minimum_swap)
    gate = {"status": "PASS" if gate_pass else "FAIL", "thresholds": {
        "minimum_relative_improvement": minimum_improvement, "minimum_context_swap_response_rmse": minimum_swap},
        "metrics": metrics,
        "interpretation": "Proceed to M3 only when the frozen context model beats the matched context-free executor and responds to context swaps."}
    metric_payload = {"seed": seed, "counts": {"train_tasks": len(train_tasks), "validation_tasks": len(val_tasks),
                                                  "train_environments": int(config["oracle"]["train_envs"]),
                                                  "validation_environments": int(config["oracle"]["validation_envs"])},
                      "validation": metrics, "learning_curve": history,
                                                      "parameter_counts": {
                                                          "context_model": sum(p.numel() for p in model.parameters()),
                                                          "context_free_model": sum(p.numel() for p in baseline.parameters())}}
    (output / "metrics.json").write_text(json.dumps(metric_payload, indent=2), encoding="utf-8")
    (output / "gate_report.md").write_text(
        f"# Q1 M2 gate report\n\nStatus: **{gate['status']}**.\n\n"
        f"Validation context MSE: {metrics['context_mse']:.6g}; context-free MSE: {metrics['context_free_mse']:.6g}; "
        f"relative improvement: {metrics['relative_improvement_over_context_free']:.3%}.\n\n"
        f"Context-swap response RMSE: {metrics['context_swap_response_rmse']}. Thresholds: "
        f"{minimum_improvement:.1%} relative improvement and {minimum_swap:g} response RMSE.\n\n"
        "M2 uses training-only normalization and independent support/query trajectories. The encoder and executor "
        "are frozen in `context_model.pt`; held-out test environments were not used. A failed gate blocks M3/M4.\n",
        encoding="utf-8")
    torch.save({"model": model.state_dict(), "normalizer": normalizer.as_dict(), "config": config,
                "seed": seed, "train_programs": [list(p) for p in train_programs]}, output / "context_model.pt")
    torch.save({"model": baseline.state_dict()}, output / "context_free_model.pt")
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    (output / "provenance.json").write_text(json.dumps({
        "git_commit": git_commit, "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "seed": seed, "device": str(device),
        "train_environment_count": int(config["oracle"]["train_envs"]),
        "validation_environment_count": int(config["oracle"]["validation_envs"]),
        "normalization_fit": "training support and query transitions only",
        "held_out_composition_trajectories_in_m2": False,
    }, indent=2), encoding="utf-8")
    with (output / "per_seed_metrics.csv").open("w", encoding="utf-8") as f:
        f.write("seed,context_mse,context_free_mse,relative_improvement,context_swap_response_rmse\n")
        f.write(f"{seed},{metrics['context_mse']},{metrics['context_free_mse']},"
                f"{metrics['relative_improvement_over_context_free']},{metrics['context_swap_response_rmse']}\n")
    with (output / "learning_curve.csv").open("w", encoding="utf-8") as f:
        f.write("epoch,train_mse_delta_v,context_free_train_mse_delta_v,train_nll_per_delta_v_dimension,"
                "context_free_train_nll_per_delta_v_dimension,validation_mse_delta_v,"
                "context_free_validation_mse_delta_v,validation_nll_per_delta_v_dimension,"
                "context_free_validation_nll_per_delta_v_dimension\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_mse_delta_v']},{row['context_free_train_mse_delta_v']},"
                    f"{row['train_nll_per_delta_v_dimension']},{row['context_free_train_nll_per_delta_v_dimension']},"
                    f"{row['validation_mse_delta_v']},{row['context_free_validation_mse_delta_v']},"
                    f"{row['validation_nll_per_delta_v_dimension']},"
                    f"{row['context_free_validation_nll_per_delta_v_dimension']}\n")
    return gate
