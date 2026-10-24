"""P0 semantic audit of saved Q1 checkpoints; this script never trains models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.datasets import make_split_manifest, sample_environment, stage_pair
from q1.evaluate import (_rank_true_program, _score_program_prediction,
                        _support_tensor, _public_transitions)
from q1.model import TransitionNormalizer
from q1.oracle import functional_classes
from q1.primitives import PrimitiveBank
from q1.simulator import PROGRAMS
from q1.train_primitives import _functional_distances, _get_m2, _make_transfer_tasks, _prepare


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bank(path: Path, context_dim: int, hidden_dim: int, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    bank = PrimitiveBank(context_dim, hidden_dim=hidden_dim).to(device)
    bank.load_state_dict(checkpoint["primitives"])
    bank.eval()
    for parameter in bank.parameters():
        parameter.requires_grad_(False)
    return bank, checkpoint


def _assert_fresh_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty repair audit directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    _assert_fresh_dir(run_dir)
    paths = {"m2": Path(args.m2_checkpoint), "m3": Path(args.m3_checkpoint),
             "m4": Path(args.m4_checkpoint)}
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Required {name.upper()} checkpoint is missing: {path}")
    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(args.threads)))
    model, normalizer, m2_data = _get_m2(paths["m2"], device)
    m3_bank, m3_data = _load_bank(paths["m3"], int(m2_data["config"]["model"]["context_dim"]),
                                  int(config["model"]["hidden_dim"] // 2), device)
    m4_bank, m4_data = _load_bank(paths["m4"], int(m2_data["config"]["model"]["context_dim"]),
                                  int(config["model"]["hidden_dim"] // 2), device)
    code_map = m3_data["group_to_code"]
    if code_map != m4_data["group_to_code"] or sorted(code_map.values()) != list(range(4)):
        raise SystemExit("M3/M4 opaque group-to-code maps are inconsistent or not bijective")

    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest_path = paths["m4"].parent.parent / "split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = make_split_manifest(int(config["seed"]), split_cfg["counts"])
    seed = int(config["seed"])
    common = {"seed": seed, "n_support": 96, "n_query": 96,
              "dt": float(config["dt"]), "sigma": float(config["noise_sigma"]),
              "parameter_ranges": split_cfg["parameter_ranges"]}
    val_tasks = _make_transfer_tasks(manifest, "validation", int(config["oracle"]["validation_envs"]),
                                     [(i,) for i in range(4)], **common)
    prepared = _prepare(val_tasks, model, normalizer, device)
    m3_distances = _functional_distances(prepared, m3_bank, model.executor, normalizer,
                                         seed=seed + 31, device=device)
    m4_distances = _functional_distances(prepared, m4_bank, model.executor, normalizer,
                                         seed=seed + 31, device=device)
    for distances in (m3_distances, m4_distances):
        distances["common_probe_rmse_delta_v"] = distances["common_probe_rmse_delta_v"].tolist()
        distances["validation_visited_rmse_delta_v"] = distances["validation_visited_rmse_delta_v"].tolist()

    budgets = (5, 10, 20, 50, 100)
    pairs = [tuple(pair) for pair in split_cfg["composition_held_out_pairs"]]
    oracle_rows = []
    for env_idx in range(int(config["oracle"]["validation_envs"])):
        params = sample_environment(manifest, "validation", env_idx, split_cfg["parameter_ranges"])
        for p_idx, program in enumerate(pairs):
            _, adaptation, _ = stage_pair(
                params, program, seed=seed + 810001 + env_idx * 1009 + p_idx * 97,
                n_support=96, n_query=max(budgets), dt=float(config["dt"]),
                sigma=float(config["noise_sigma"]))
            rows = _public_transitions(adaptation.states, adaptation.actions)
            classes, _ = functional_classes(params)
            for n in budgets:
                predicted = _rank_true_program(params, rows[:n], float(config["dt"]))
                score = _score_program_prediction(params, program, predicted)
                oracle_rows.append({"environment": env_idx, "true_program": str(program), "budget": n,
                                    "predicted_program": str(predicted), **score})

    oracle_summary = {}
    for budget in budgets:
        rows = [row for row in oracle_rows if row["budget"] == budget]
        oracle_summary[str(budget)] = {
            "physical_oracle_program_identity_accuracy": float(np.mean([r["identity_correct"] for r in rows])),
            "physical_oracle_functional_class_accuracy": float(np.mean([r["functional_class_correct"] for r in rows])),
            "case_count": len(rows),
        }
    payload = {
        "audit_kind": "P0_NO_TRAINING_DEVELOPMENT_AUDIT",
        "m3_validation_functional_distance_rmse_delta_v": m3_distances,
        "m4_validation_functional_distance_rmse_delta_v": m4_distances,
        "functional_distance_gate_probe_set": "common_fixed_validation_probe_set",
        "physical_oracle_same_stage_pair_summary": oracle_summary,
        "physical_oracle_stage_pair_details": oracle_rows,
        "group_to_code": code_map,
        "program_enumeration": [list(program) for program in PROGRAMS],
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (run_dir / "diagnostic_tables.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(oracle_rows[0]))
        writer.writeheader()
        writer.writerows(oracle_rows)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoints = {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                   for name, path in paths.items()}
    (run_dir / "provenance.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "training_performed": False, "checkpoint_inputs": checkpoints,
        "source_split_manifest": str(manifest_path.resolve()),
        "validation_split_only": True,
        "physical_oracle_uses_same_stage_pair_adaptation_trajectory": True,
        "runtime_seconds": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")
    (run_dir / "gate_report.md").write_text(
        "# Q1 P0 repair audit\n\n"
        "This is a no-training development audit of saved M2, M3, and M4 checkpoints. "
        "M3/M4 functional distances are in physical delta-v units on both common fixed validation probes "
        "and naturally visited validation state-action pairs; the gate-compatible probe set is the common set. "
        "Physical oracle rankings use the same uninterrupted stage-pair adaptation transitions as the program "
        "scorer and report raw identity separately from functional class. See `metrics.json`, "
        "`diagnostic_tables.csv`, and `provenance.json`.\n",
        encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--m2-checkpoint", required=True)
    parser.add_argument("--m3-checkpoint", required=True)
    parser.add_argument("--m4-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": "COMPLETE_NO_TRAINING", "run_dir": args.run_dir,
                      "oracle": result["physical_oracle_same_stage_pair_summary"]}, indent=2))


if __name__ == "__main__":
    main()
