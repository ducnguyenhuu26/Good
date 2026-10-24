"""Train the preregistered after-only vs paired-effect M3 development ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.model import TransitionNormalizer
from q1.primitives import PrimitiveBank
from q1.simulator import PROGRAMS, response
from q1.train_primitives import (_fit_single_operators_paired, _functional_distances,
                                 _get_m2, _make_transfer_tasks, _paired_effect_scales, _prepare)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _eval_effect_bank(prepared, bank, executor, normalizer, code_map, *, dt, device,
                      common_probes):
    by_code = {code: [] for code in range(4)}
    probe_sets = {"natural_validation": [], "common_fixed_validation": []}
    for task, context, x_norm, _ in prepared:
        code = task.group
        code_id = int(code_map[str(code)])
        natural_raw = task.query[:, :6]
        for set_name, raw_x in (("natural_validation", natural_raw),
                                ("common_fixed_validation", common_probes)):
            x_raw = torch.as_tensor(raw_x, dtype=torch.float32, device=device)
            x_normed = normalizer.normalize_x(x_raw)
            target_before = response(task.before_params, raw_x, dt=dt)
            target_after = response(task.target_params, raw_x, dt=dt)
            target_effect = target_after - target_before
            y_after = torch.as_tensor(target_after, dtype=torch.float32, device=device)
            y_effect = torch.as_tensor(target_effect, dtype=torch.float32, device=device)
            pred_before = normalizer.denormalize_dv(executor(x_normed, context.expand(len(x_normed), -1)))
            pred_after = normalizer.denormalize_dv(executor(
                x_normed, bank.apply_one(context, code_id).expand(len(x_normed), -1)))
            effect_pred = pred_after - pred_before
            persistence_mse = float(torch.mean((pred_before - y_after) ** 2).cpu())
            after_mse = float(torch.mean((pred_after - y_after) ** 2).cpu())
            effect_mse = float(torch.mean((effect_pred - y_effect) ** 2).cpu())
            effect_energy = float(torch.mean(y_effect ** 2).cpu())
            wrong_mse = []
            for wrong_code in range(4):
                if wrong_code == code_id:
                    continue
                prediction = normalizer.denormalize_dv(executor(
                    x_normed, bank.apply_one(context, wrong_code).expand(len(x_normed), -1)))
                wrong_mse.append(float(torch.mean((prediction - y_after) ** 2).cpu()))
            row = {"environment": task.env_id, "physical_intervention": code,
                   "effect_signal_energy": effect_energy, "effect_mse": effect_mse,
                   "effect_skill_vs_zero_effect": (1.0 - effect_mse / effect_energy
                                                    if effect_energy > 1e-12 else None),
                   "after_dynamics_mse": after_mse, "persistence_after_mse": persistence_mse,
                   "relative_after_mse_gain_vs_persistence": (persistence_mse - after_mse)
                   / max(persistence_mse, 1e-12), "wrong_code_after_mse": float(np.mean(wrong_mse)),
                   "true_code_after_mse": after_mse, "probe_set": set_name}
            by_code[code].append(row)
            probe_sets[set_name].append(row)
    return by_code


def _summary(per_code_rows, threshold_skill, threshold_after_gain):
    result = {}
    for code, rows in per_code_rows.items():
        summary = {}
        for key in ("effect_signal_energy", "effect_mse", "effect_skill_vs_zero_effect",
                    "after_dynamics_mse", "persistence_after_mse",
                    "relative_after_mse_gain_vs_persistence", "wrong_code_after_mse"):
            values = [float(row[key]) for row in rows if row[key] is not None]
            mean = float(np.mean(values))
            half_width = (1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
                          if len(values) > 1 else 0.0)
            summary[key] = {"mean": mean, "ci95_normal_approx": [mean - half_width, mean + half_width],
                            "environment_count": len(values)}
        summary["gate"] = {
            "effect_skill_pass": summary["effect_skill_vs_zero_effect"]["mean"] >= threshold_skill,
            "after_gain_pass": summary["relative_after_mse_gain_vs_persistence"]["mean"] >= threshold_after_gain,
        }
        summary["gate"]["pass"] = summary["gate"]["effect_skill_pass"] and summary["gate"]["after_gain_pass"]
        result[str(code)] = summary
    return result


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    out = Path(args.run_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty repair M3 directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config_path, m2_path, m3_path = Path(args.config), Path(args.m2_checkpoint), Path(args.m3_checkpoint)
    for path, name in ((m2_path, "M2"), (m3_path, "M3 reference")):
        if not path.is_file():
            raise SystemExit(f"Required {name} checkpoint is missing: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(args.threads)))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    m2_model, normalizer, m2_data = _get_m2(m2_path, device)
    m3_reference = torch.load(m3_path, map_location="cpu", weights_only=False)
    code_map = m3_reference["group_to_code"]
    if sorted(code_map.values()) != list(range(4)):
        raise SystemExit("M3 checkpoint contains a non-bijective group-to-code map")
    manifest_path = m2_path.parent.parent / "split_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"The M2 run split manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    common = {"seed": seed, "n_support": 96, "n_query": 96, "dt": float(config["dt"]),
              "sigma": float(config["noise_sigma"]), "parameter_ranges": split_cfg["parameter_ranges"]}
    single_programs = [(i,) for i in range(4)]
    train_tasks = _make_transfer_tasks(manifest, "train", int(config["oracle"]["train_envs"]),
                                       single_programs, **common)
    val_tasks = _make_transfer_tasks(manifest, "validation", int(config["oracle"]["validation_envs"]),
                                     single_programs, **common)
    train_prepared = _prepare(train_tasks, m2_model, normalizer, device)
    val_prepared = _prepare(val_tasks, m2_model, normalizer, device)
    scales = _paired_effect_scales(
        train_prepared, dt=float(config["dt"]),
        scale_floor_fraction=float(config["training"]["effect_scale_floor_fraction"]))
    betas = [0.0] + [float(x) for x in config["training"]["effect_beta_candidates"]]
    init_seed = seed + 51007
    torch.manual_seed(init_seed)
    initial_bank = PrimitiveBank(int(m2_data["config"]["model"]["context_dim"]),
                                 hidden_dim=int(config["model"]["hidden_dim"] // 2)).to(device)
    initial_state = deepcopy(initial_bank.state_dict())
    fixed_probes = np.concatenate((
        np.random.default_rng(seed + 4801).uniform(-0.6, 0.6, (512, 4)),
        np.random.default_rng(seed + 4802).uniform(-1.0, 1.0, (512, 2))), axis=1)

    variant_results = {}
    all_rows = []
    for beta in betas:
        variant = f"beta_{beta:.2f}"
        variant_dir = out / variant
        variant_dir.mkdir()
        bank = PrimitiveBank(int(m2_data["config"]["model"]["context_dim"]),
                             hidden_dim=int(config["model"]["hidden_dim"] // 2)).to(device)
        bank.load_state_dict(initial_state)
        # Reset each variant to the same optimizer RNG stream and identical sample sequence.
        history = _fit_single_operators_paired(
            train_prepared, bank, m2_model.executor, normalizer, code_map, scales,
            dt=float(config["dt"]), beta=beta,
            epochs=int(config["training"]["primitive_epochs"]), seed=seed,
            learning_rate=float(config["training"]["learning_rate"]), device=device,
            backgrounds_per_batch=int(config["training"]["backgrounds_per_batch"]),
            queries_per_task=int(config["training"]["queries_per_task"]),
            residual_regularization=float(config["training"].get("effect_residual_regularization", 0.0)))
        rows_by_code = _eval_effect_bank(val_prepared, bank, m2_model.executor, normalizer,
                                         code_map, dt=float(config["dt"]), device=device,
                                         common_probes=fixed_probes)
        threshold_skill = float(config["gates"]["minimum_effect_skill_per_code"])
        threshold_after = float(config["gates"]["minimum_after_mse_relative_gain_per_code"])
        natural_rows = {code: [row for row in rows if row["probe_set"] == "natural_validation"]
                        for code, rows in rows_by_code.items()}
        summary = _summary(natural_rows, threshold_skill, threshold_after)
        gate = all(item["gate"]["pass"] for item in summary.values())
        for group_rows in rows_by_code.values():
            all_rows.extend([{"beta": beta, **row} for row in group_rows])
        variant_results[variant] = {"beta": beta, "status": "PASS" if gate else "FAIL",
                                    "per_code_validation": summary,
                                    "fixed_probe_rows": sum(row["probe_set"] == "common_fixed_validation"
                                                            for values in rows_by_code.values() for row in values),
                                    "natural_probe_rows": sum(row["probe_set"] == "natural_validation"
                                                              for values in rows_by_code.values() for row in values)}
        (variant_dir / "metrics.json").write_text(json.dumps(variant_results[variant], indent=2), encoding="utf-8")
        (variant_dir / "learning_curve.csv").write_text(
            "epoch,normalized_after_mse_loss,balanced_normalized_effect_mse_loss,primitive_residual_mse,beta\n" +
            "".join(f"{r['epoch']},{r['normalized_after_mse_loss']},{r['balanced_normalized_effect_mse_loss']},"
                    f"{r['primitive_residual_mse']},{r['beta']}\n"
                    for r in history), encoding="utf-8")
        torch.save({"primitives": bank.state_dict(), "group_to_code": code_map,
                    "beta": beta, "seed": seed, "initialization_seed": init_seed,
                    "training_scales": scales}, variant_dir / "primitives.pt")
    # Select beta only from validation metrics; beta=0 remains the fair after-only primary control.
    candidates = {key: value for key, value in variant_results.items() if value["beta"] > 0}
    selected_variant = max(candidates, key=lambda key: (
        min(v["effect_skill_vs_zero_effect"]["mean"] for v in candidates[key]["per_code_validation"].values()),
        min(v["relative_after_mse_gain_vs_persistence"]["mean"]
            for v in candidates[key]["per_code_validation"].values())))
    variants_json = {"run_kind": "TRAIN_SPLIT_ONLY_M3_EFFECT_ABLATION_DEVELOPMENT",
                     "selected_beta_variant_by_validation": selected_variant,
                     "selection_metric": config["selection"], "variants": variant_results,
                     "training_scales": scales,
                     "m3_gate_thresholds": config["gates"],
                     "validation_gate_status": {k: v["status"] for k, v in variant_results.items()}}
    (out / "metrics.json").write_text(json.dumps(variants_json, indent=2), encoding="utf-8")
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (out / "per_code_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    unique_counterfactual_pairs = sum(len(task.query) for task in train_tasks)
    training_counterfactual_response_evals = (2 * len(betas)
        * int(config["training"]["primitive_epochs"])
        * len(train_tasks) * int(config["training"]["queries_per_task"]))
    scale_counterfactual_response_evals = 2 * unique_counterfactual_pairs
    (out / "provenance.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "threads": torch.get_num_threads(), "seed": seed,
        "m2_checkpoint": {"path": str(m2_path.resolve()), "sha256": _sha256(m2_path)},
        "m3_reference_checkpoint": {"path": str(m3_path.resolve()), "sha256": _sha256(m3_path)},
        "split_manifest": str(manifest_path.resolve()), "training_split_only": True,
        "validation_used_for_beta_selection": True, "test_split_used": False,
        "training_supervision": "paired simulator-only conditional means on identical train state-action probes",
        "unique_train_counterfactual_probe_count": unique_counterfactual_pairs,
        "counterfactual_simulator_response_evaluations": (
            training_counterfactual_response_evals + scale_counterfactual_response_evals),
        "train_background_count": int(config["oracle"]["train_envs"]),
        "optimizer_updates_per_variant": int(config["training"]["primitive_epochs"])
            * int(np.ceil(config["oracle"]["train_envs"] / config["training"]["backgrounds_per_batch"])),
        "fair_ablation": "identical initialization, code map, examples, updates, minibatch seeds, and optimizer",
        "runtime_seconds": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")
    (out / "gate_report.md").write_text(
        "# Q1 paired-effect M3 ablation\n\n"
        "This is a development training run using training-background simulator counterfactuals. "
        "The after-only beta=0 control and effect variants share initialization, code map, state-action examples, "
        "optimizer updates, and minibatch seeds. Beta selection uses validation only. No test environments were used.\n\n"
        f"Validation-selected beta: {variant_results[selected_variant]['beta']:.2f}. Variant gate statuses: "
        + ", ".join(f"{key}={value['status']}" for key, value in variant_results.items())
        + ". See per-code CIs in `metrics.json` and probe-level rows in `per_code_metrics.csv`.\n",
        encoding="utf-8")
    return variants_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--m2-checkpoint", required=True)
    parser.add_argument("--m3-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"run_dir": args.run_dir,
                      "selected_beta_variant_by_validation": result["selected_beta_variant_by_validation"],
                      "validation_gate_status": result["validation_gate_status"]}, indent=2))


if __name__ == "__main__":
    main()
