from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.datasets import make_split_manifest, sample_environment
from q1.oracle import oracle_discrimination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q1 M1 simulator and oracle checks")
    parser.add_argument("--config", default="configs/q1_smoke.yaml")
    parser.add_argument("--output", default="results/q1/m1_smoke")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_config = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(config["seed"])
    oracle_config = config["oracle"]
    exploration = config["exploration"]
    manifest = make_split_manifest(seed, split_config["counts"])
    backgrounds = [sample_environment(manifest, oracle_config["split"], i,
                                      split_config["parameter_ranges"])
                   for i in range(int(oracle_config.get("backgrounds", 1)))]
    oracle = oracle_discrimination(
        backgrounds, seed=seed + 101,
        budgets=tuple(oracle_config["budgets"]),
        cases_per_program=int(oracle_config["cases_per_program"]),
        dt=float(config["dt"]), sigma=float(config["noise_sigma"]),
        initial_q_std=float(exploration["initial_q_std"]),
        initial_v_std=float(exploration["initial_v_std"]),
        action_limit=float(exploration["action_limit"]),
        action_signal_scale=float(exploration["action_signal_scale"]),
        action_jitter=float(exploration["action_jitter"]),
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oracle_metrics.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps({
        "git_commit": git_commit, "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch_version, "seed": seed,
        "split": oracle_config["split"], "background_count": len(backgrounds),
        "cases_per_program": int(oracle_config["cases_per_program"]),
        "random_streams": {"parameter": "environment_seeds in split_manifest.json",
                           "initial_state": "case_seed", "actions": "case_seed+104729",
                           "dynamics_noise": "case_seed+209759"},
    }, indent=2), encoding="utf-8")
    with (out / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "budget", "identity_accuracy", "functional_class_accuracy",
                         "mean_best_second_margin_mse"])
        for n, m in oracle["budgets"].items():
            writer.writerow([seed, n, m["identity_accuracy"], m["functional_class_accuracy"],
                             m["mean_best_second_margin_mse"]])
    report = [
        "# Q1 M1 gate report", "", f"Run kind: **{config['run_kind']}**", "",
        f"Python: {platform.python_version()}; NumPy: {np.__version__}; PyTorch: {torch_version}.",
        "", f"Dynamics use dt={config['dt']}, action limits [-{exploration['action_limit']},"
        f"{exploration['action_limit']}], initial q~N(0,{exploration['initial_q_std']}²), "
        f"v~N(0,{exploration['initial_v_std']}²), "
        f"and sigma={config['noise_sigma']} velocity noise. Oracle programs are scored against the same "
        "continuous adaptation trajectory for each budget.", "", "## Oracle discrimination", "",
        "| N | Identity accuracy | Functional class accuracy | Mean best-second MSE margin | U1→U2 class acc. | U2→U1 class acc. |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n, metrics in oracle["budgets"].items():
        orders = metrics["noncommuting_order_class_accuracy"]
        report.append(f"| {n} | {metrics['identity_accuracy']:.3f} | {metrics['functional_class_accuracy']:.3f} | "
                      f"{metrics['mean_best_second_margin_mse']:.3g} | {orders['(U1,U2)']:.3f} | "
                      f"{orders['(U2,U1)']:.3f} |")
    report.extend([
        "", f"Functional equivalence used {oracle['probe_count']} fixed state-action probes and max-response "
        f"tolerance {oracle['equivalence_tolerance']}. The U1→U2 / U2→U1 response RMSE is "
        f"{oracle['noncommuting_probe_rmse_u1_u2_vs_u2_u1']:.6g}.", "",
        "## Decision", "",
        f"This run is **{config['run_kind']}**. Smoke results are not scientific evidence. This validation-only "
        "development diagnostic does not use final-test backgrounds or select a checkpoint. Neural stages remain gated on M1.",
    ])
    (out / "gate_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote M1 diagnostics to {out.resolve()}")


if __name__ == "__main__":
    main()
