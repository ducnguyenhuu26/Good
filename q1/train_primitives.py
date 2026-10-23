"""M3 primitive transfer and M4 composition training/evaluation."""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from q1.datasets import rollout, stage_pair
from q1.evaluate import choose_length_penalty, evaluate_regime
from q1.inference import rank_programs
from q1.model import ContextDynamicsModel, TransitionNormalizer
from q1.primitives import DirectUpdateBank, PrimitiveBank, map_intervention_program
from q1.simulator import PROGRAMS, DynamicsParams, apply_program, sample_params
from q1.train_dynamics import _transitions


@dataclass
class TransferTask:
    support: np.ndarray
    query: np.ndarray
    before_params: DynamicsParams
    target_params: DynamicsParams
    group: int
    program: tuple[int, ...]
    env_id: str


def _get_m2(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]["model"]
    model = ContextDynamicsModel(int(cfg["context_dim"]), int(cfg["hidden_dim"]),
                                 depth=int(cfg.get("depth", 2)),
                                 dropout=float(cfg.get("dropout", 0.0))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalizer = TransitionNormalizer.from_dict(checkpoint["normalizer"])
    return model, normalizer, checkpoint


def _make_transfer_tasks(manifest: dict, split: str, count: int, programs: list[tuple[int, ...]],
                         *, seed: int, n_support: int, n_query: int, dt: float, sigma: float,
                         parameter_ranges: dict) -> list[TransferTask]:
    tasks = []
    for env_idx in range(count):
        env_id = manifest["environment_ids"][split][env_idx]
        base = sample_params(np.random.default_rng(manifest["environment_seeds"][split][env_idx]),
                             manifest["regions"][split], parameter_ranges)
        for p_idx, program in enumerate(programs):
            after = apply_program(base, program)
            run_seed = seed + env_idx * 100003 + p_idx * 7919
            support = _transitions(rollout(base, n=n_support, seed=run_seed, dt=dt, sigma=sigma))
            query = _transitions(rollout(after, n=n_query, seed=run_seed + 17, dt=dt, sigma=sigma))
            tasks.append(TransferTask(support, query, base, after,
                                      program[0] if len(program) == 1 else -1, program, env_id))
    return tasks


def _prepare(tasks: list[TransferTask], model: ContextDynamicsModel,
             normalizer: TransitionNormalizer, device: torch.device):
    prepared = []
    with torch.no_grad():
        for task in tasks:
            support = torch.as_tensor(task.support, dtype=torch.float32, device=device)
            support = torch.cat((normalizer.normalize_x(support[:, :6]),
                                 normalizer.normalize_dv(support[:, 6:])), dim=-1)[None]
            context = model.encode(support)
            query = torch.as_tensor(task.query, dtype=torch.float32, device=device)
            prepared.append((task, context, normalizer.normalize_x(query[:, :6]),
                             normalizer.normalize_dv(query[:, 6:])))
    return prepared


def _fit_single_operators(prepared, bank: PrimitiveBank, direct: DirectUpdateBank,
                          executor, normalizer: TransitionNormalizer,
                          predictive_variance: float, code_map: dict[str, int], epochs: int, seed: int,
                          learning_rate: float, device: torch.device,
                          backgrounds_per_batch: int = 4, queries_per_task: int = 8):
    optimizer = torch.optim.Adam(bank.parameters(), lr=learning_rate)
    direct_optimizer = torch.optim.Adam(direct.parameters(), lr=learning_rate)
    by_group: dict[int, list] = {i: [] for i in range(4)}
    for item in prepared:
        by_group[item[0].group].append(item)
    env_count = min(len(v) for v in by_group.values())
    history = []
    rng = np.random.default_rng(seed + 407)
    for epoch in range(epochs):
        order = rng.permutation(env_count)
        losses, direct_losses = [], []
        for start in range(0, env_count, backgrounds_per_batch):
            env_ids = order[start:start + backgrounds_per_batch]
            contexts, codes, xs, ys = [], [], [], []
            for group in range(4):
                selected = [by_group[group][int(i)] for i in env_ids]
                for task, context, x, y in selected:
                    idx = rng.choice(len(x), size=min(queries_per_task, len(x)), replace=False)
                    contexts.extend([context[0]] * len(idx))
                    codes.extend([code_map[str(group)]] * len(idx))
                    xs.extend(x[idx])
                    ys.extend(y[idx])
            c = torch.stack(contexts)
            code_tensor = torch.tensor(codes, dtype=torch.long, device=device)
            x = torch.stack(xs)
            y = torch.stack(ys)
            transformed = bank(c, code_tensor)
            pred_actual = normalizer.denormalize_dv(executor(x, transformed))
            target_actual = normalizer.denormalize_dv(y)
            loss = torch.mean(0.5 * (np.log(2.0 * np.pi * predictive_variance)
                                     + (pred_actual - target_actual) ** 2 / predictive_variance))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            direct_contexts = direct(c, code_tensor)
            direct_actual = normalizer.denormalize_dv(executor(x, direct_contexts))
            direct_loss = torch.mean(0.5 * (np.log(2.0 * np.pi * predictive_variance)
                                            + (direct_actual - target_actual) ** 2 / predictive_variance))
            direct_optimizer.zero_grad(set_to_none=True)
            direct_loss.backward()
            direct_optimizer.step()
            direct_losses.append(float(direct_loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "single_operator_nll_per_delta_v_dimension": float(np.mean(losses)),
                        "direct_update_nll_per_delta_v_dimension": float(np.mean(direct_losses))})
    return history


@torch.no_grad()
def _eval_single(prepared, bank: PrimitiveBank, direct: DirectUpdateBank,
                 code_map: dict[str, int], executor, normalizer: TransitionNormalizer,
                 predictive_variance: float, device: torch.device):
    metrics = {}
    for group in range(4):
        op_errors, persistence, wrong, direct_errors = [], [], [], []
        op_actual_mse, op_nll = [], []
        for task, context, x, y in [p for p in prepared if p[0].group == group]:
            code = code_map[str(group)]
            pred = executor(x, bank.apply_one(context, code))
            op_errors.append(float(torch.mean((pred - y) ** 2).cpu()))
            pred_actual = normalizer.denormalize_dv(pred)
            truth_actual = normalizer.denormalize_dv(y)
            residual = pred_actual - truth_actual
            op_actual_mse.append(float(torch.mean(residual ** 2).cpu()))
            nll = 0.5 * (np.log(2.0 * np.pi * predictive_variance)
                         + residual ** 2 / predictive_variance)
            op_nll.append(float(torch.mean(nll).cpu()))
            persistence.append(float(torch.mean((executor(x, context) - y) ** 2).cpu()))
            wrong_code_errors = [float(torch.mean((executor(x, bank.apply_one(context, c)) - y) ** 2).cpu())
                                 for c in range(4) if c != code]
            wrong.append(float(np.mean(wrong_code_errors)))
            direct_errors.append(float(torch.mean((executor(x, direct.apply_one(context, code)) - y) ** 2).cpu()))
        metrics[str(group)] = {"operator_mse": float(np.mean(op_errors)),
                               "operator_mse_delta_v": float(np.mean(op_actual_mse)),
                               "operator_nll_per_delta_v_dimension": float(np.mean(op_nll)),
                               "persistence_mse": float(np.mean(persistence)),
                               "wrong_code_mse": float(np.mean(wrong)),
                               "direct_update_mse": float(np.mean(direct_errors))}
    return metrics


@torch.no_grad()
def _functional_distances(prepared, bank: PrimitiveBank, executor, *, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    probes = np.concatenate((rng.uniform(-0.6, 0.6, (256, 4)), rng.uniform(-1, 1, (256, 2))), axis=1)
    distances = np.zeros((4, 4), dtype=np.float64)
    values = []
    for task, context, _, _ in prepared[:min(16, len(prepared))]:
        x = torch.as_tensor(probes, dtype=torch.float32, device=device)
        responses = []
        for code in range(4):
            transformed = bank.apply_one(context, code)
            responses.append(executor(x, transformed.expand(len(x), -1)).cpu().numpy())
        values.append(responses)
    for i in range(4):
        for j in range(4):
            distances[i, j] = float(np.sqrt(np.mean([
                np.mean((item[i] - item[j]) ** 2) for item in values
            ])))
    return distances


def run_m3(config: dict, output_dir: str | Path, m2_checkpoint: str | Path,
           *, seed: int | None = None, device_name: str = "cpu") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"] if seed is None else seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((output.parent / "split_manifest.json").read_text(encoding="utf-8"))
    m2, normalizer, checkpoint = _get_m2(m2_checkpoint, device)
    n_support, n_query = (64, 64) if config["run_kind"] == "SMOKE" else (96, 96)
    common = dict(seed=seed, n_support=n_support, n_query=n_query, dt=float(config["dt"]),
                  sigma=float(config["noise_sigma"]), parameter_ranges=split_cfg["parameter_ranges"])
    train_tasks = _make_transfer_tasks(manifest, "train", int(config["oracle"]["train_envs"]),
                                       [(i,) for i in range(4)], **common)
    val_tasks = _make_transfer_tasks(manifest, "validation", int(config["oracle"]["validation_envs"]),
                                     [(i,) for i in range(4)], **common)
    train_prepared, val_prepared = _prepare(train_tasks, m2, normalizer, device), _prepare(val_tasks, m2, normalizer, device)
    code_order = list(range(4))
    random.Random(seed + 1777).shuffle(code_order)
    group_to_code = {str(group): code for group, code in enumerate(code_order)}
    bank = PrimitiveBank(int(checkpoint["config"]["model"]["context_dim"]), hidden_dim=int(config["model"]["hidden_dim"] // 2)).to(device)
    direct = DirectUpdateBank(int(checkpoint["config"]["model"]["context_dim"]), hidden_dim=int(config["model"]["hidden_dim"] // 2)).to(device)
    history = _fit_single_operators(train_prepared, bank, direct, m2.executor, normalizer,
                                    float(config["model"]["predictive_variance"]), group_to_code,
                                    int(config["training"]["primitive_epochs"]), seed,
                                    float(config["training"]["learning_rate"]), device)
    bank_metrics = _eval_single(val_prepared, bank, direct, group_to_code, m2.executor,
                                normalizer, float(config["model"]["predictive_variance"]), device)
    distances = _functional_distances(val_prepared, bank, m2.executor, seed=seed + 31, device=device)
    avg_op = float(np.mean([v["operator_mse"] for v in bank_metrics.values()]))
    avg_persist = float(np.mean([v["persistence_mse"] for v in bank_metrics.values()]))
    avg_wrong = float(np.mean([v["wrong_code_mse"] for v in bank_metrics.values()]))
    min_distance = float(np.min(distances[np.triu_indices(4, 1)]))
    gain = min((avg_persist - avg_op) / max(avg_persist, 1e-12),
               (avg_wrong - avg_op) / max(avg_wrong, 1e-12))
    gates = config["gates"]
    gate_pass = (gain >= float(gates["m3_min_relative_gain"])
                 and min_distance >= float(gates["m3_min_functional_distance"]))
    metrics = {"per_code": bank_metrics, "average_operator_mse": avg_op,
               "average_persistence_mse": avg_persist, "average_wrong_code_mse": avg_wrong,
               "minimum_relative_gain_over_controls": gain,
               "functional_distance_rmse": distances.tolist(), "minimum_pairwise_functional_rmse": min_distance}
    (output / "metrics.json").write_text(json.dumps({"seed": seed,
        "counts": {"train_transfer_tasks": len(train_tasks), "validation_transfer_tasks": len(val_tasks)},
        "group_to_code": group_to_code, "validation": metrics, "learning_curve": history}, indent=2), encoding="utf-8")
    (output / "gate_report.md").write_text(
        f"# Q1 M3 gate report\n\nStatus: **{'PASS' if gate_pass else 'FAIL'}**.\n\n"
        f"Mean operator MSE: {avg_op:.6g}; persistence: {avg_persist:.6g}; wrong-code: {avg_wrong:.6g}; "
        f"minimum relative gain over both controls: {gain:.3%}.\n\n"
        f"Minimum pairwise functional response RMSE: {min_distance:.6g}.\n\n"
        "Training uses opaque same-intervention groups with a fixed random code bijection saved in the checkpoint. "
        "M2 encoder/executor weights remain frozen. A failed gate blocks M4.\n", encoding="utf-8")
    torch.save({"primitives": bank.state_dict(), "direct_update": direct.state_dict(),
                "group_to_code": group_to_code, "seed": seed, "m2_checkpoint": str(m2_checkpoint)},
               output / "primitives_m3.pt")
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    (output / "provenance.json").write_text(json.dumps({"git_commit": git_commit,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "seed": seed, "device": str(device), "training_supervision": "opaque intervention group plus known training group order",
        "m2_parameters_frozen": True, "group_to_code": group_to_code}, indent=2), encoding="utf-8")
    with (output / "per_seed_metrics.csv").open("w", encoding="utf-8") as f:
        f.write("seed,average_operator_mse,average_persistence_mse,average_wrong_code_mse,minimum_relative_gain\n")
        f.write(f"{seed},{avg_op},{avg_persist},{avg_wrong},{gain}\n")
    with (output / "learning_curve.csv").open("w", encoding="utf-8") as f:
        f.write("epoch,single_operator_nll_per_delta_v_dimension,direct_update_nll_per_delta_v_dimension\n")
        for row in history:
            f.write(f"{row['epoch']},{row['single_operator_nll_per_delta_v_dimension']},"
                    f"{row['direct_update_nll_per_delta_v_dimension']}\n")
    return {"status": "PASS" if gate_pass else "FAIL", "metrics": metrics}


def _fit_m4(prepared, bank: PrimitiveBank, executor, code_map: dict[str, int],
            normalizer: TransitionNormalizer, predictive_variance: float,
            epochs: int, seed: int, learning_rate: float, device: torch.device,
            batch_tasks: int = 32, queries_per_task: int = 8):
    optimizer = torch.optim.Adam(bank.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed + 901)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(prepared))
        losses = []
        for start in range(0, len(order), batch_tasks):
            selected = [prepared[int(i)] for i in order[start:start + batch_tasks]]
            contexts, xs, ys = [], [], []
            for task, context, x, y in selected:
                code_program = map_intervention_program(task.program, code_map)
                transformed = bank.apply_program(context, code_program)
                idx = rng.choice(len(x), size=min(queries_per_task, len(x)), replace=False)
                contexts.extend([transformed[0]] * len(idx))
                xs.extend(x[idx])
                ys.extend(y[idx])
            c, x, y = torch.stack(contexts), torch.stack(xs), torch.stack(ys)
            prediction = normalizer.denormalize_dv(executor(x, c))
            target = normalizer.denormalize_dv(y)
            loss = torch.mean(0.5 * (np.log(2.0 * np.pi * predictive_variance)
                                     + (prediction - target) ** 2 / predictive_variance))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "composition_finetune_nll_per_delta_v_dimension": float(np.mean(losses))})
    return history


def run_m4(config: dict, output_dir: str | Path, m2_checkpoint: str | Path,
           m3_checkpoint: str | Path, *, seed: int | None = None,
           device_name: str = "cpu") -> dict:
    from q1.evaluate import _public_transitions

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"] if seed is None else seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((output.parent / "split_manifest.json").read_text(encoding="utf-8"))
    m2, normalizer, m2_data = _get_m2(m2_checkpoint, device)
    m3_data = torch.load(m3_checkpoint, map_location=device, weights_only=False)
    code_map = m3_data["group_to_code"]
    context_dim = int(m2_data["config"]["model"]["context_dim"])
    bank = PrimitiveBank(context_dim, hidden_dim=int(config["model"]["hidden_dim"] // 2)).to(device)
    bank.load_state_dict(m3_data["primitives"])
    # During M4, only the shared primitive network and its code embeddings move.
    train_envs = int(config["oracle"]["train_envs"])
    n_support, n_query = (64, 64) if config["run_kind"] == "SMOKE" else (96, 96)
    common = dict(seed=seed, n_support=n_support, n_query=n_query,
                  dt=float(config["dt"]), sigma=float(config["noise_sigma"]),
                  parameter_ranges=split_cfg["parameter_ranges"])
    train_programs = [tuple(i) for i in ([(g,) for g in range(4)] + split_cfg["composition_training_pairs"])]
    train_tasks = _make_transfer_tasks(manifest, "train", train_envs, train_programs, **common)
    train_prepared = _prepare(train_tasks, m2, normalizer, device)
    history = _fit_m4(train_prepared, bank, m2.executor, code_map, normalizer,
                      float(config["model"]["predictive_variance"]),
                      int(config["training"]["primitive_epochs"]), seed,
                      float(config["training"]["learning_rate"]), device)

    # Select lambda_length only on validation backgrounds and permitted composition programs.
    val_envs = int(config["oracle"]["validation_envs"])
    val_programs = [tuple(i) for i in split_cfg["composition_training_pairs"]]
    val_params = [sample_params(np.random.default_rng(manifest["environment_seeds"]["validation"][i]),
                                "validation", split_cfg["parameter_ranges"])
                  for i in range(val_envs)]
    validation_cases = []
    max_adapt = 20
    for env_idx, params in enumerate(val_params):
        for p_idx, program in enumerate(val_programs):
            old, adaptation, _ = stage_pair(params, program, seed=seed + 700001 + env_idx * 1009,
                                            n_support=int(config.get("old_context_support", 96)),
                                            n_query=max_adapt, dt=float(config["dt"]), sigma=float(config["noise_sigma"]))
            validation_cases.append((params, old, adaptation, program))
    length_candidates = config.get("inference", {}).get("length_penalty_candidates", [0.0, 0.1, 1.0, 10.0])
    predictive_variance = float(config["model"]["predictive_variance"])
    length_penalty = choose_length_penalty(validation_cases, m2, bank, normalizer, code_map, device,
                                          candidates=length_candidates,
                                          predictive_variance=predictive_variance)

    test_env_count = int(config["oracle"]["test_envs"])
    budgets = [5, 10, 20, 50, 100]
    regimes = {
        "id": ("test_id", [()] + [(i,) for i in range(4)] + val_programs),
        "composition_ood": ("test_id", [tuple(p) for p in split_cfg["composition_held_out_pairs"]]),
        "background_ood": ("test_background_ood", val_programs),
        "joint_ood": ("test_background_ood", [tuple(p) for p in split_cfg["composition_held_out_pairs"]]),
    }
    test_cases = []
    for regime, (split, programs) in regimes.items():
        for env_idx in range(test_env_count):
            params = sample_params(np.random.default_rng(manifest["environment_seeds"][split][env_idx]),
                                   split, split_cfg["parameter_ranges"])
            for p_idx, program in enumerate(programs):
                seed_offset = (0 if split == "test_id" else 500000)
                old, adaptation, query = stage_pair(
                    params, program, seed=seed + seed_offset + env_idx * 1009,
                    n_support=int(config.get("old_context_support", 96)), n_query=max(budgets),
                    dt=float(config["dt"]), sigma=float(config["noise_sigma"]))
                for budget in budgets:
                    test_cases.append((regime, params, old, adaptation, query, program, budget))
    metrics = evaluate_regime(test_cases, m2, bank, normalizer, code_map, length_penalty, device,
                              direct_steps=int(config["training"].get("direct_context_steps", 40)),
                              direct_regularization=float(config["training"].get("direct_context_regularization", 1e-3)),
                              predictive_variance=predictive_variance, dt=float(config["dt"]))
    # Gate on held-out functional transfer and paired query-MSE gain over direct context adaptation.
    composition_rows = [v for k, v in metrics.items() if k.startswith("composition_ood|") or k.startswith("joint_ood|")]
    direct_gains = []
    class_scores = []
    for row in composition_rows:
        direct_gains.append((row["direct_context_query_mse"] - row["learned_search_query_mse"])
                            / max(row["direct_context_query_mse"], 1e-12))
        class_scores.append(row["class_accuracy"])
    best_gain = max(direct_gains, default=float("-inf"))
    mean_ood_class_accuracy = float(np.mean(class_scores)) if class_scores else 0.0
    thresholds = config["gates"]
    gate_pass = (best_gain >= float(thresholds["m4_min_relative_gain_vs_direct"])
                 and mean_ood_class_accuracy >= float(thresholds["m4_min_ood_class_accuracy"]))
    result = {"seed": seed, "old_context_support_budget": int(config.get("old_context_support", 96)),
              "candidate_program_count": len(PROGRAMS),
              "length_penalty": length_penalty, "validation_length_penalty_candidates": length_candidates,
              "regimes": metrics, "m4_gate": {"status": "PASS" if gate_pass else "FAIL",
              "best_relative_query_mse_gain_vs_direct": best_gain,
              "mean_composition_ood_class_accuracy": mean_ood_class_accuracy,
              "thresholds": {"minimum_relative_gain_vs_direct": thresholds["m4_min_relative_gain_vs_direct"],
                             "minimum_ood_class_accuracy": thresholds["m4_min_ood_class_accuracy"]}}}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "gate_report.md").write_text(
        f"# Q1 M4 gate report\n\nStatus: **{'PASS' if gate_pass else 'FAIL'}**.\n\n"
        f"Validation-selected length penalty: {length_penalty:g}. Best relative fresh-query MSE gain over "
        f"direct old-context-conditioned adaptation in held-out composition regimes: {best_gain:.3%}. "
        f"Mean composition-OOD functional-class accuracy: {mean_ood_class_accuracy:.3f}.\n\n"
        "Regimes report ID, composition OOD, background OOD, and joint OOD separately. Program identity metrics "
        "are accompanied by functional-class accuracy. One CPU seed is development evidence only; use the "
        "confirmatory multi-seed config before making a scientific claim.\n", encoding="utf-8")
    torch.save({"primitives": bank.state_dict(), "group_to_code": code_map,
                "length_penalty": length_penalty, "m2_checkpoint": str(m2_checkpoint),
                "m3_checkpoint": str(m3_checkpoint), "seed": seed}, output / "primitives_m4.pt")
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    (output / "provenance.json").write_text(json.dumps({"git_commit": git_commit,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "seed": seed, "device": str(device), "candidate_program_count": len(PROGRAMS),
        "old_context_support_budget": int(config.get("old_context_support", 96)),
        "length_penalty_selected_on": "validation split and permitted training compositions",
        "test_composition_tuning": False}, indent=2), encoding="utf-8")
    with (output / "per_seed_metrics.csv").open("w", encoding="utf-8") as f:
        f.write("seed,regime,budget,sample_count,identity_accuracy,class_accuracy,learned_query_mse,direct_query_mse\n")
        for key, row in metrics.items():
            regime, n_text = key.split("|N=")
            f.write(f"{seed},{regime},{n_text},{row['count']},{row['identity_accuracy']},"
                    f"{row['class_accuracy']},{row['learned_search_query_mse']},{row['direct_context_query_mse']}\n")
    with (output / "learning_curve.csv").open("w", encoding="utf-8") as f:
        f.write("epoch,composition_finetune_nll_per_delta_v_dimension\n")
        for row in history:
            f.write(f"{row['epoch']},{row['composition_finetune_nll_per_delta_v_dimension']}\n")
    return result["m4_gate"]
