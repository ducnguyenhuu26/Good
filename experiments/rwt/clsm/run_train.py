"""Reference initialization CLI; full CLSM training remains behind the release gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import gymnasium
import mujoco
import numpy as np
import scipy
import torch

from rwt.clsm.config import load_config
from rwt.clsm.factory import build_clsm
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.train import train_reference
from experiments.rwt.clsm.build_datasets import _sha256
from experiments.rwt.clsm.collectors import SmoothActionCollector
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld


def collect_reference(env_id: str, n: int, seed: int):
    schema = mujoco_nominal_schema(env_id)
    obs_rows, actions, next_rows = [], [], []
    world = 0
    while len(obs_rows) < n:
        env = HiddenPhysicsWorld(env_id, [HiddenLaw(0)] * min(500, n - len(obs_rows)), schema)
        try:
            obs, _ = env.reset(seed=seed + world)
            collector = SmoothActionCollector(schema.action_low, schema.action_high, seed + 10000 + world)
            for _ in range(len(env.schedule)):
                action = collector.act(obs)
                next_obs, _, terminated, truncated, _ = env.step(action)
                obs_rows.append(obs.copy()); actions.append(action.copy()); next_rows.append(next_obs.copy())
                obs = next_obs
                if terminated or truncated:
                    break
        finally:
            env.close()
        world += 1
    return np.asarray(obs_rows), np.asarray(actions), np.asarray(next_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--track", required=True, choices=("comp","evol","mem"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["track"] != args.track:
        parser.error("config track and --track disagree")
    if args.dry_run:
        print(json.dumps({"env": args.env, "track": args.track, "device": args.device,
                          "reference_only": args.reference_only, "config": config}, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in installed PyTorch")
    if args.resume or args.evaluate_only:
        parser.error("resume/evaluate-only require full CLSM checkpoints; the M-step and checkpoint selection are not implemented")
    if not args.reference_only:
        parser.error("full CLSM training blocked: the joint generalized-EM M-step, validation checkpoint selection, and full experiment protocol are not implemented; reference-only is not full CLSM")
    seed = config["seed"] if args.seed is None else args.seed
    output = args.output_dir / args.track / args.env / "reference_only" / str(seed)
    data_path = output / "reference_nominal.npz"
    if args.check_data:
        if not data_path.exists():
            parser.error(f"missing nominal reference data: {data_path}")
        print(data_path)
        return
    output.mkdir(parents=True, exist_ok=True)
    n = config["transitions"]["train"] if config.get("mode") == "smoke" else 10000
    obs, action, next_obs = collect_reference(args.env, n, seed)
    np.savez_compressed(data_path, obs=obs, action=action, next_obs=next_obs)
    torch.manual_seed(seed)
    model = build_clsm(mujoco_nominal_schema(args.env), config).law
    curve = train_reference(model, torch.from_numpy(obs), torch.from_numpy(action), torch.from_numpy(next_obs),
                            epochs=config["training"]["reference_epochs"],
                            lr=config["training"]["lr_dynamics"], seed=seed)
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__}
    commit = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    checkpoint = output / "checkpoint_reference.pt"
    torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed,
                "data_sha256": _sha256(data_path), "versions": versions, "git_commit": commit,
                "torch_rng_state": torch.get_rng_state()}, checkpoint)
    report = {"status": "reference_initialization_only", "train_loss": curve,
              "checkpoint": str(checkpoint), "data_sha256": _sha256(data_path),
              "checkpoint_sha256": _sha256(checkpoint), "versions": versions,
              "git_commit": commit}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checkpoint": str(checkpoint), "final_loss": curve[-1]}))


if __name__ == "__main__":
    main()
