"""Fit a nominal SAC collection policy using unmodified public Gym observations.

This collector policy is separate from CLSM's stationary public reward and is
never used for checkpoint selection or the core model likelihood.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import gymnasium as gym

from experiments.rwt.clsm.build_datasets import _sha256


def fit(env_id: str, steps: int, seed: int, output_dir: Path) -> dict:
    if steps < 1:
        raise ValueError("positive nominal controller budget required")
    try:
        import stable_baselines3 as sb3
    except ImportError as error:
        raise RuntimeError("install the rwt-clsm-controller extra for nominal SAC collection") from error
    run_dir = output_dir / env_id / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id)
    try:
        controller = sb3.SAC("MlpPolicy", env, seed=seed, verbose=0,
                             learning_starts=min(1000, max(1, steps // 10)),
                             device="cpu")
        controller.learn(total_timesteps=steps, progress_bar=False)
        path = run_dir / "nominal_sac.zip"
        controller.save(path)
    finally:
        env.close()
    report = {"status": "nominal_collection_controller", "env": env_id,
              "seed": seed, "timesteps": steps, "policy_file": str(path),
              "policy_sha256": _sha256(path), "native_gym_reward_used_for_collector_only": True,
              "python": platform.python_version(), "gymnasium": gym.__version__,
              "stable_baselines3": sb3.__version__}
    (run_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm/controllers"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"env": args.env, "steps": args.steps, "seed": args.seed}))
        return
    print(json.dumps(fit(args.env, args.steps, args.seed, args.output_dir)))


if __name__ == "__main__":
    main()
