from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.datasets import make_split_manifest
from q1.train_dynamics import run_m2
from q1.train_primitives import run_m3, run_m4


STAGES = {"m1": 1, "m2": 2, "m3": 3, "m4": 4}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gated Q1 pipeline")
    parser.add_argument("--config", default="configs/q1_smoke.yaml")
    parser.add_argument("--through", choices=STAGES, default="m2")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true", help="Reuse completed stage artifacts and run missing stages")
    parser.add_argument("--reuse-m2-checkpoint", help="Load an existing frozen M2 checkpoint from this path")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    seed = int(config["seed"])
    run_dir = Path(args.run_dir or f"results/q1/{config['run_kind'].lower()}_seed{seed}")
    resolved_config_path = run_dir / "config.resolved.yaml"
    if run_dir.exists():
        if not args.resume:
            raise SystemExit(f"Run directory exists: {run_dir}. Use --resume to reuse its artifacts or choose --run-dir.")
        if resolved_config_path.exists():
            previous_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
            if previous_config != config:
                raise SystemExit("Resolved config differs from this run. Choose a new --run-dir for changed settings.")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest = make_split_manifest(seed, split_cfg["counts"])
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    m1_dir = run_dir / "m1"
    if not ((m1_dir / "metrics.json").exists() and args.resume):
        test_cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_q1_simulator.py",
                    "tests/test_q1_splits.py", "tests/test_q1_model_contracts.py",
                    "tests/test_q1_no_leakage.py"]
        subprocess.run(test_cmd, check=True)
        subprocess.run([sys.executable, "scripts/q1_m1_check.py", "--config", args.config,
                        "--output", str(m1_dir), "--seed", str(seed)], check=True)
    m1 = json.loads((m1_dir / "metrics.json").read_text(encoding="utf-8"))
    if m1["m1_oracle_gate"]["status"] != "PASS":
        raise SystemExit("M1 oracle gate failed; pipeline stopped before M2.")
    if STAGES[args.through] == 1:
        print(f"Completed through M1: {run_dir}")
        return

    m2_dir = run_dir / "m2"
    reuse_checkpoint = False
    if args.reuse_m2_checkpoint:
        checkpoint = Path(args.reuse_m2_checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(f"Requested M2 checkpoint does not exist: {checkpoint}")
        m2_dir.mkdir(parents=True, exist_ok=True)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("config") != config or int(saved.get("seed", -1)) != seed:
            raise SystemExit("M2 checkpoint config/seed do not match this run; refusing an untracked reuse.")
        source_report = checkpoint.parent / "gate_report.md"
        if not source_report.is_file() or "Status: **PASS**" not in source_report.read_text(encoding="utf-8"):
            raise SystemExit("Checkpoint reuse requires an adjacent passing M2 gate report.")
        source_manifest = checkpoint.parent.parent / "split_manifest.json"
        if not source_manifest.is_file() or json.loads(source_manifest.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("M2 checkpoint split manifest does not match the current run.")
        shutil.copy2(checkpoint, m2_dir / "context_model.pt")
        for artifact in ("metrics.json", "gate_report.md", "context_free_model.pt", "learning_curve.csv"):
            source = checkpoint.parent / artifact
            if source.is_file():
                shutil.copy2(source, m2_dir / artifact)
        reuse_checkpoint = True
    if reuse_checkpoint or (args.resume and (m2_dir / "context_model.pt").exists() and (m2_dir / "gate_report.md").exists()):
        gate_text = (m2_dir / "gate_report.md").read_text(encoding="utf-8")
        m2_pass = "Status: **PASS**" in gate_text
    else:
        gate = run_m2(config, m2_dir, seed=seed, device_name=args.device)
        m2_pass = gate["status"] == "PASS"
    if STAGES[args.through] == 2:
        print(f"Completed through M2 (gate {'PASS' if m2_pass else 'FAIL'}): {run_dir}")
        return
    if not m2_pass:
        raise SystemExit("M2 gate failed; pipeline stopped before M3.")
    m3_dir = run_dir / "m3"
    if args.resume and (m3_dir / "primitives_m3.pt").exists() and (m3_dir / "gate_report.md").exists():
        m3_pass = "Status: **PASS**" in (m3_dir / "gate_report.md").read_text(encoding="utf-8")
    else:
        m3_result = run_m3(config, m3_dir, m2_dir / "context_model.pt", seed=seed, device_name=args.device)
        m3_pass = m3_result["status"] == "PASS"
    if STAGES[args.through] == 3:
        print(f"Completed through M3 (gate {'PASS' if m3_pass else 'FAIL'}): {run_dir}")
        return
    if not m3_pass:
        raise SystemExit("M3 gate failed; pipeline stopped before M4.")
    m4_dir = run_dir / "m4"
    if args.resume and (m4_dir / "primitives_m4.pt").exists() and (m4_dir / "gate_report.md").exists():
        m4_pass = "Status: **PASS**" in (m4_dir / "gate_report.md").read_text(encoding="utf-8")
    else:
        m4_result = run_m4(config, m4_dir, m2_dir / "context_model.pt", m3_dir / "primitives_m3.pt",
                           seed=seed, device_name=args.device)
        m4_pass = m4_result["status"] == "PASS"
    print(f"Completed through M4 (gate {'PASS' if m4_pass else 'FAIL'}): {run_dir}")


if __name__ == "__main__":
    main()
