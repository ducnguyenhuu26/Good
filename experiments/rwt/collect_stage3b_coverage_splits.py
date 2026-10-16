"""Collect separate A/B/D locomotion coverage splits from valid controllers.

The split is by disjoint episode seeds and trajectories. C is not collected.
These files are a coverage diagnostic and are not added to Pilot A training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from rwt.stage3a import CEMConfig
from collect_stage3b_coverage import collect_regime, load_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/rwt/configs/stage3b/coverage_splits.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if set(config["splits"]) != {"train", "validation", "test"}:
        raise ValueError("coverage collection requires train/validation/test splits")
    if set(config["regimes"]) != {"A", "B", "D"}:
        raise ValueError("coverage collection is restricted to A/B/D")
    split_seeds = {int(value) for value in config["splits"].values()}
    if any(abs(left - right) <= 25000
           for index, left in enumerate(sorted(split_seeds))
           for right in sorted(split_seeds)[index + 1:]):
        raise ValueError("coverage split seed ranges overlap")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(config["checkpoint_dir"])
    world = load_world(checkpoints, int(config["dynamics_seed"]),
                       int(config["reward_seed"]))
    source_hashes = {
        name: hashlib.sha256((checkpoints / filename).read_bytes()).hexdigest()
        for name, filename in (
            ("dynamics", f"pooled_{config['dynamics_seed']}_best.pt"),
            ("reward", f"reward_{config['reward_seed']}_best.pt"),
        )
    }
    metadata = []
    for split, seed in config["splits"].items():
        for regime_index, regime in enumerate(("A", "B", "D")):
            arrays, row = collect_regime(
                regime=regime,
                seed=int(seed) + regime_index * 10000,
                count_per_policy=int(config["count_per_policy"]),
                episode_steps=int(config["episode_steps"]),
                world=world,
                cem=CEMConfig(**config["mpc"]),
            )
            filename = f"{split}_{regime}.npz"
            path = output / filename
            np.savez_compressed(path, **arrays)
            row.update({
                "split": split,
                "file": filename,
                "count": len(arrays["state"]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "velocity_gt_0_5": float((arrays["state"][:, 8] > .5).mean()),
                "velocity_mean": float(arrays["state"][:, 8].mean()),
                "mean_abs_action": float(np.abs(arrays["action"]).mean()),
            })
            metadata.append(row)
    manifest = {
        "offline_regimes": ["A", "B", "D"],
        "held_out_regime": "C",
        "split_unit": "disjoint episode seeds and trajectories",
        "source_checkpoints": source_hashes,
        "config": config,
        "splits": metadata,
    }
    (output / "coverage_splits_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
