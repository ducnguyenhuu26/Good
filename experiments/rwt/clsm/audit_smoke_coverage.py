"""Evaluator-only code/event coverage audit for retained public smoke splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(results_dir: Path, evaluator_dir: Path, env_id: str) -> dict:
    output = {}
    for track in ("comp", "evol", "mem"):
        output[track] = {}
        for split in ("train", "validation", "test"):
            public_path = results_dir / track / env_id / "data" / f"{split}.npz"
            oracle_path = evaluator_dir / track / env_id / f"{split}_oracle.npz"
            with np.load(public_path) as public, np.load(oracle_path) as oracle:
                code = oracle["code"].astype(int)
                reset = oracle["reset"].astype(bool)
                world_id = public["world_id"]
                world_step = public["world_step"]
                if len(code) != len(world_id):
                    raise ValueError("public/oracle row counts differ")
                within_world = world_id[1:] == world_id[:-1]
                unique, counts = np.unique(code, return_counts=True)
                output[track][split] = {
                    "transitions": len(code),
                    "worlds": int(np.sum(world_step == 0)),
                    "initial_codes": code[world_step == 0].tolist(),
                    "code_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
                    "within_world_structural_switches": int(np.sum(
                        (code[1:] != code[:-1]) & within_world)),
                    "same_code_reset_labels": int(reset.sum()),
                    "code_zero_null_reset_labels": int(np.sum(reset & (code == 0))),
                }
    return {"status": "evaluator_only_coverage_diagnostic", "env": env_id,
            "tracks": output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.results_dir, args.evaluator_dir, args.env)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
