"""Prepare the locked A/B/D NPZ split without training a Stage 3A model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

if __package__:
    from .run_stage3a import _collect_one, _regimes
    from .run_stage3a1 import dataset_audit, load_split
else:
    from run_stage3a import _collect_one, _regimes
    from run_stage3a1 import dataset_audit, load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/rwt/configs/stage3a/cpu_smoke.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if tuple(config["offline"]["regimes"]) != ("A", "B", "D"):
        raise ValueError("locked Stage 3B split must use A/B/D only")
    root = Path(config["output_dir"])
    data_dir = root / "data"
    manifest_path = root / "data_manifest.json"
    if manifest_path.exists():
        train = {r: load_split(data_dir, "train", r) for r in ("A", "B", "D")}
        val = {r: load_split(data_dir, "validation", r) for r in ("A", "B", "D")}
        print(json.dumps(dataset_audit(data_dir, train, val, manifest_path), indent=2))
        return
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError("dataset directory is partial or unmanifested; inspect before collecting")
    data_dir.mkdir(parents=True, exist_ok=True)
    regimes = _regimes(config)
    rows = []
    for split, count in (("train", int(config["offline"]["train_count_per_regime"])),
                         ("validation", int(config["offline"]["validation_count_per_regime"]))):
        for index, regime in enumerate(("A", "B", "D")):
            seed = int(config["seed"]) + (0 if split == "train" else 100000) + 1000 * index
            rows.append(_collect_one(regime, count, int(config["offline"]["episode_steps"]),
                                     seed, regimes, data_dir / f"{split}_{regime}.npz"))
    manifest = {"offline_regimes": ["A", "B", "D"], "held_out_regime": "C",
                "splits": rows,
                "dataset_fields": ["state", "action", "next_state", "reward", "terminated", "truncated"]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train = {r: load_split(data_dir, "train", r) for r in ("A", "B", "D")}
    val = {r: load_split(data_dir, "validation", r) for r in ("A", "B", "D")}
    print(json.dumps(dataset_audit(data_dir, train, val, manifest_path), indent=2))


if __name__ == "__main__":
    main()
