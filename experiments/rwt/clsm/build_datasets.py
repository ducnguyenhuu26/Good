"""Track-separated MuJoCo dataset builder with evaluator-only oracle files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from rwt.clsm.config import load_config
from rwt.clsm.schemas import mujoco_nominal_schema
from experiments.rwt.clsm.collectors import SmoothActionCollector
from experiments.rwt.clsm.physics_wrappers import HiddenPhysicsWorld
from experiments.rwt.clsm.world_schedule import make_schedule


PUBLIC_KEYS = ("obs", "action", "reward", "next_obs", "terminated", "truncated", "world_id", "world_step")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(env_id: str, track: str, config: dict, output_dir: Path) -> dict:
    if track not in ("comp", "evol", "mem"):
        raise ValueError("invalid track")
    if config["track"] != track:
        raise ValueError("CLI track and config track disagree")
    if config.get("collector") != "smooth_random_smoke_v1" or config.get("mode") != "smoke":
        raise RuntimeError("full collection blocked: predeclared nominal controller mixture is not implemented; use cpu_smoke.yaml")
    schema = mujoco_nominal_schema(env_id)
    public_root = output_dir / track / env_id / "data"
    evaluator_root = output_dir.parent / "clsm_evaluator" / track / env_id
    public_root.mkdir(parents=True, exist_ok=True)
    evaluator_root.mkdir(parents=True, exist_ok=True)
    split_manifest = {}
    evaluator_manifest = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        count = config["transitions"][split]
        arrays = {k: [] for k in PUBLIC_KEYS}
        hidden = {"code": [], "motor_1": [], "motor_2": [], "damping": [], "reset": []}
        world_ids = []
        world_number = 0
        while len(arrays["reward"]) < count:
            world_id = split_index * 100000 + world_number
            world_ids.append(world_id)
            remaining = count - len(arrays["reward"])
            length = min(config["world_length"], remaining)
            schedule = make_schedule(track, split, length, config["seed"] + 100000 * split_index + world_number)
            env = HiddenPhysicsWorld(env_id, schedule, schema)
            try:
                obs, _ = env.reset(seed=config["seed"] + 200000 * split_index + world_number)
                collector = SmoothActionCollector(schema.action_low, schema.action_high,
                                                  config["seed"] + 300000 * split_index + world_number)
                for t in range(length):
                    action = collector.act(obs)
                    current = env.law_at
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    if not np.isfinite(next_obs).all() or not np.isfinite(reward):
                        raise RuntimeError(f"nonfinite transition in {split} world {world_id} step {t}")
                    for key, value in zip(PUBLIC_KEYS, (obs.copy(), action.copy(), reward,
                                                          next_obs.copy(), terminated, truncated, world_id, t)):
                        arrays[key].append(value)
                    for key in hidden:
                        hidden[key].append(getattr(current, key))
                    obs = next_obs
                    if terminated or truncated:
                        break
            finally:
                env.close()
            world_number += 1
            if world_number > count:
                raise RuntimeError("collection cannot produce enough finite transitions")
        public_path = public_root / f"{split}.npz"
        oracle_path = evaluator_root / f"{split}_oracle.npz"
        np.savez_compressed(public_path, **{key: np.asarray(value) for key, value in arrays.items()})
        np.savez_compressed(oracle_path, **{key: np.asarray(value) for key, value in hidden.items()})
        split_manifest[split] = {"public_file": str(public_path), "public_sha256": _sha256(public_path),
                                  "transitions": len(arrays["reward"]), "world_ids": world_ids,
                                  "collector_id": SmoothActionCollector.policy_id}
        evaluator_manifest[split] = {"evaluator_file": str(oracle_path),
                                     "evaluator_sha256": _sha256(oracle_path),
                                     "codes_present": sorted(set(hidden["code"]))}
    if set(split_manifest["train"]["world_ids"]) & set(split_manifest["test"]["world_ids"]):
        raise AssertionError("split world overlap")
    manifest = {"mode": "smoke", "track": track, "env": env_id, "seed": config["seed"],
                "schema": {"obs_dim": schema.observation_dim, "act_dim": schema.action_dim,
                            "velocity_indices": schema.joint_velocity_indices,
                            "controlled_dofs": schema.controlled_dof_indices},
                "splits": split_manifest}
    (public_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (evaluator_root / "manifest.json").write_text(json.dumps(evaluator_manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--track", required=True, choices=("comp", "evol", "mem"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["track"] != args.track:
        raise SystemExit("config track and --track disagree")
    if args.dry_run:
        print(json.dumps({"env": args.env, "track": args.track, "config": config}, indent=2))
        return
    if args.check_data:
        path = args.output_dir / args.track / args.env / "data" / "manifest.json"
        if not path.exists():
            raise SystemExit(f"missing dataset manifest: {path}")
        print(path.read_text(encoding="utf-8"))
        return
    result = build(args.env, args.track, config, args.output_dir)
    print(json.dumps({"track": result["track"], "env": result["env"],
                      "counts": {k: v["transitions"] for k, v in result["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
