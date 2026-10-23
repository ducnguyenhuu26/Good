from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the declared Q1 confirmatory seeds")
    parser.add_argument("--config", default="configs/q1_confirmatory.yaml")
    parser.add_argument("--through", choices=("m1", "m2", "m3", "m4"), default="m4")
    parser.add_argument("--output-root", default="results/q1/confirmatory")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for seed in config["seed_list"]:
        run_dir = root / f"seed_{seed}"
        command = [sys.executable, "scripts/q1_pipeline.py", "--config", args.config,
                   "--through", args.through, "--seed", str(seed), "--run-dir", str(run_dir)]
        if args.resume:
            command.append("--resume")
        process = subprocess.run(command, check=False)
        record = {"seed": seed, "exit_code": process.returncode,
                  "run_dir": str(run_dir), "stage_status": {}}
        for stage in ("m1", "m2", "m3", "m4"):
            report = run_dir / stage / "gate_report.md"
            if report.exists():
                report_text = report.read_text(encoding="utf-8")
                record["stage_status"][stage] = next(
                    ("PASS" if "Status: **PASS**" in line else "FAIL" for line in report_text.splitlines()
                     if line.startswith("Status:")), "UNKNOWN")
        outcomes.append(record)
        (root / "seed_results.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    summary = {"seed_results": outcomes, "complete_five_seed_interval": False,
               "paired_method_deltas": {},
               "interval_method": "two-sided Student t interval, df=4, t*=2.776; direct-context minus learned-search MSE (positive favors learned)"}
    m4_paths = [Path(row["run_dir"]) / "m4" / "metrics.json" for row in outcomes]
    if len(m4_paths) == 5 and all(path.is_file() for path in m4_paths):
        loaded = [json.loads(path.read_text(encoding="utf-8")) for path in m4_paths]
        key_sets = [set(obj["regimes"]) for obj in loaded]
        if all(keys == key_sets[0] for keys in key_sets):
            summary["complete_five_seed_interval"] = True
            for key in sorted(key_sets[0]):
                deltas = [obj["regimes"][key]["direct_context_query_mse"]
                          - obj["regimes"][key]["learned_search_query_mse"] for obj in loaded]
                values = np.asarray(deltas, dtype=np.float64)
                half_width = 2.776 * float(values.std(ddof=1)) / np.sqrt(len(values))
                summary["paired_method_deltas"][key] = {
                    "per_seed_direct_minus_learned_mse": deltas,
                    "mean_direct_minus_learned_mse": float(values.mean()),
                    "ci95": [float(values.mean() - half_width), float(values.mean() + half_width)],
                }
    (root / "confirmatory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print((root / "confirmatory_summary.json").resolve())


if __name__ == "__main__":
    main()
