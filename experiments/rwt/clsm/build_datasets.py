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
from experiments.rwt.clsm.collectors import (SmoothActionCollector, NominalControllerMixture,
                                              R2StationaryExcitationCollector)
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld
from experiments.rwt.clsm.world_schedule import make_schedule


PUBLIC_KEYS = ("obs", "action", "reward", "next_obs", "terminated", "truncated", "world_id", "world_step")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(env_id: str, track: str, config: dict, output_dir: Path,
          controller_checkpoint: Path | None = None) -> dict:
    if track not in ("comp", "evol", "mem"):
        raise ValueError("invalid track")
    if config["track"] != track:
        raise ValueError("CLI track and config track disagree")
    mode = config.get("mode")
    if mode == "smoke":
        if config.get("collector") != SmoothActionCollector.policy_id:
            raise ValueError("smoke collector ID mismatch")
        collector_class = SmoothActionCollector
        controller = None
    elif mode == "full":
        if config.get("collector") != "nominal_controller_mixture_required":
            raise ValueError("full collection requires registered nominal policy mixture")
        if controller_checkpoint is None or not controller_checkpoint.is_file():
            raise FileNotFoundError("full collection requires --controller-checkpoint from nominal SAC fit")
        controller_manifest_path = controller_checkpoint.parent / "manifest.json"
        if not controller_manifest_path.is_file():
            raise FileNotFoundError("nominal controller provenance manifest missing")
        controller_manifest = json.loads(controller_manifest_path.read_text(encoding="utf-8"))
        if (controller_manifest.get("env") != env_id or
                controller_manifest.get("policy_sha256") != _sha256(controller_checkpoint) or
                controller_manifest.get("timesteps", 0) < config.get("collector_min_nominal_steps", 100000)):
            raise ValueError("nominal controller provenance, hash, or training budget invalid")
        try:
            from stable_baselines3 import SAC
        except ImportError as error:
            raise RuntimeError("install rwt-clsm-controller extra for full collection") from error
        controller = SAC.load(controller_checkpoint, device="cpu")
        collector_class = NominalControllerMixture
    elif mode == "pilot":
        if config.get("collector") != R2StationaryExcitationCollector.policy_id or track != "comp":
            raise ValueError("R2 pilot requires its registered COMP stationary excitation collector")
        collector_class = R2StationaryExcitationCollector
        controller = None
    else:
        raise ValueError("mode must be smoke, pilot or full")
    schema = mujoco_nominal_schema(env_id)
    if controller is not None and (controller.observation_space.shape != (schema.observation_dim,)
                                   or controller.action_space.shape != (schema.action_dim,)):
        raise ValueError("nominal controller action/observation schema mismatch")
    public_root = output_dir / track / env_id / "data"
    evaluator_root = output_dir.parent / "clsm_evaluator" / track / env_id
    previous_manifest = public_root / "manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        expected_condition = config.get("schedule_condition", "stationary_calibration" if mode == "pilot" else "in_family")
        if (previous.get("mode") != mode or
                previous.get("schedule_condition", "in_family") != expected_condition):
            raise RuntimeError("existing dataset has another mode or schedule; choose an isolated --output-dir")
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
            if mode == "pilot":
                # Balanced stationary nominal/single-mechanism calibration worlds.
                code = (0, 1, 2, 4)[world_number % 4]
                strength_rng = np.random.default_rng(config["seed"] + 900000 * split_index + world_number)
                motor_1, motor_2 = strength_rng.uniform(.50, .85, 2)
                damping = strength_rng.uniform(1.30, 2.00)
                law = HiddenLaw(code, float(motor_1), float(motor_2), float(damping), False)
                schedule = [law] * length
            else:
                schedule = make_schedule(track, split, length, config["seed"] + 100000 * split_index + world_number,
                                         condition=config.get("schedule_condition", "in_family"))
            env = HiddenPhysicsWorld(env_id, schedule, schema)
            try:
                obs, _ = env.reset(seed=config["seed"] + 200000 * split_index + world_number)
                collector_seed = config["seed"] + 300000 * split_index + world_number
                collector = (collector_class(schema.action_low, schema.action_high, collector_seed)
                             if controller is None else
                             collector_class(schema.action_low, schema.action_high, controller, collector_seed))
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
        action_values = np.asarray(arrays["action"])
        observation_values = np.asarray(arrays["obs"])
        velocity_values = observation_values[:, list(schema.joint_velocity_indices)]
        coverage = {"action_quantiles_10_50_90": np.quantile(action_values, [.1, .5, .9], axis=0).tolist(),
                    "joint_velocity_quantiles_10_50_90": np.quantile(velocity_values, [.1, .5, .9], axis=0).tolist(),
                    "observation_norm_quantiles_10_50_90": np.quantile(
                        np.linalg.norm(observation_values, axis=1), [.1, .5, .9]).tolist(),
                    "finite_transition_fraction": 1.0}
        split_manifest[split] = {"public_file": str(public_path), "public_sha256": _sha256(public_path),
                                  "transitions": len(arrays["reward"]), "world_ids": world_ids,
                                  "collector_id": collector_class.policy_id, "coverage": coverage}
        evaluator_manifest[split] = {"evaluator_file": str(oracle_path),
                                     "evaluator_sha256": _sha256(oracle_path),
                                     "codes_present": sorted(set(hidden["code"]))}
    if set(split_manifest["train"]["world_ids"]) & set(split_manifest["test"]["world_ids"]):
        raise AssertionError("split world overlap")
    manifest = {"mode": mode, "track": track, "env": env_id, "seed": config["seed"],
                "schedule_condition": config.get("schedule_condition", "stationary_calibration" if mode == "pilot" else "in_family"),
                "pilot_design": ({"codes": [0, 1, 2, 4], "stationary_worlds": True,
                                  "strength_bounds": {"motor": [.50, .85], "damping": [1.30, 2.00]},
                                  "labels_location": "evaluator-only"} if mode == "pilot" else None),
                "controller_sha256": _sha256(controller_checkpoint) if controller_checkpoint else None,
                "schema": {"obs_dim": schema.observation_dim, "act_dim": schema.action_dim,
                            "velocity_indices": schema.joint_velocity_indices,
                            "controlled_dofs": schema.controlled_dof_indices},
                "resolved_config": config,
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
    parser.add_argument("--controller-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["track"] != args.track:
        raise SystemExit("config track and --track disagree")
    if config.get("environment_family") == "maniskill_pd":
        raise SystemExit("PushCube-v1 PD adapter has not passed a local ManiSkill controller/physics audit")
    if args.dry_run:
        print(json.dumps({"env": args.env, "track": args.track, "config": config}, indent=2))
        return
    if args.check_data:
        path = args.output_dir / args.track / args.env / "data" / "manifest.json"
        if not path.exists():
            raise SystemExit(f"missing dataset manifest: {path}")
        print(path.read_text(encoding="utf-8"))
        return
    result = build(args.env, args.track, config, args.output_dir, args.controller_checkpoint)
    print(json.dumps({"track": result["track"], "env": result["env"],
                      "counts": {k: v["transitions"] for k, v in result["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
