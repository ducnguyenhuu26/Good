"""Collect bounded A/B/D-only train addendum for the Stage 3B pilot.

The controller consumes observations only. Regime identity is collection-side
metadata and never appears in the transition NPZ or controller input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from rwt.stage3a import CEMConfig, MiniCEM, RecurrentHalfCheetah
from rwt.stage3b import RewardPredictor, SeparatedWorldModel, StateDynamics


def load_world(checkpoint_dir: Path, dynamics_seed: int, reward_seed: int) -> SeparatedWorldModel:
    dynamics_path = checkpoint_dir / f"pooled_{dynamics_seed}_best.pt"
    reward_path = checkpoint_dir / f"reward_{reward_seed}_best.pt"
    dynamics_payload = torch.load(dynamics_path, map_location="cpu", weights_only=True)
    reward_payload = torch.load(reward_path, map_location="cpu", weights_only=True)
    if (dynamics_payload["offline_regimes"] != ["A", "B", "D"]
            or reward_payload["offline_regimes"] != ["A", "B", "D"]):
        raise ValueError("coverage controller checkpoint is not A/B/D-only")
    dynamics = StateDynamics()
    dynamics.load_state_dict(dynamics_payload["model_state"])
    reward = RewardPredictor()
    reward.load_state_dict(reward_payload["model_state"])
    return SeparatedWorldModel(dynamics, reward).eval()


def collect_regime(regime: str, seed: int, count_per_policy: int,
                   episode_steps: int, world: SeparatedWorldModel,
                   cem: CEMConfig) -> tuple[dict[str, np.ndarray], dict]:
    if regime not in ("A", "B", "D") or count_per_policy < 1 or episode_steps < 1:
        raise ValueError("coverage addendum permits positive A/B/D collection only")
    rows = {"state": [], "action": [], "next_state": [], "reward": [],
            "terminated": [], "truncated": []}
    policy_counts, policy_episodes = {}, {}
    for policy_index, policy in enumerate(("oscillator", "learned_mpc")):
        collected, episode = 0, 0
        while collected < count_per_policy:
            episode_seed = seed + 1000 * policy_index + episode
            rng = np.random.default_rng(episode_seed)
            env = RecurrentHalfCheetah([(regime, episode_steps)])
            try:
                state, _ = env.reset(seed=episode_seed)
                planner = MiniCEM(world, env.action_space.low, env.action_space.high,
                                  cem, seed=episode_seed)
                phase = rng.uniform(-np.pi, np.pi)
                for step in range(min(episode_steps, count_per_policy - collected)):
                    if policy == "oscillator":
                        action = 0.8 * np.sin(0.3 * step + np.arange(6) * np.pi / 3 + phase)
                    else:
                        action = planner.act(state)
                    if not env.action_space.contains(action.astype(env.action_space.dtype)):
                        raise AssertionError("coverage controller generated out-of-bounds action")
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    if not np.isfinite(next_state).all() or not np.isfinite(reward):
                        raise FloatingPointError("nonfinite coverage transition")
                    rows["state"].append(state.copy())
                    rows["action"].append(action.copy())
                    rows["next_state"].append(next_state.copy())
                    rows["reward"].append(float(reward))
                    rows["terminated"].append(bool(terminated))
                    # The last retained row marks a boundary even if the
                    # requested count ends before the simulator schedule.
                    rows["truncated"].append(bool(truncated or step + 1 ==
                                                  min(episode_steps, count_per_policy - collected)))
                    state = next_state
                collected += min(episode_steps, count_per_policy - collected)
                episode += 1
            finally:
                env.close()
        policy_counts[policy] = collected
        policy_episodes[policy] = episode
    arrays = {key: np.asarray(value, dtype=np.bool_ if key in ("terminated", "truncated")
                              else np.float32) for key, value in rows.items()}
    return arrays, {"regime_evaluator_only": regime, "seed": seed,
                    "policy_counts": policy_counts, "episodes_per_policy": policy_episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("experiments/rwt/configs/stage3b/coverage_cpu.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["checkpoint_dir"])
    world = load_world(checkpoint_dir, int(config["dynamics_seed"]), int(config["reward_seed"]))
    metadata = []
    for index, regime in enumerate(("A", "B", "D")):
        arrays, row = collect_regime(regime, int(config["seed"]) + index * 10000,
                                     int(config["count_per_policy"]), int(config["episode_steps"]),
                                     world, CEMConfig(**config["mpc"]))
        path = output / f"train_extra_{regime}.npz"
        np.savez_compressed(path, **arrays)
        row.update({"file": str(path), "count": len(arrays["state"]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "velocity_gt_0_5": float((arrays["state"][:, 8] > .5).mean()),
                    "mean_abs_action": float(np.abs(arrays["action"]).mean())})
        metadata.append(row)
    manifest = {"offline_regimes": ["A", "B", "D"], "held_out_regime": "C",
                "split": "train_only", "source_checkpoints": {
                    "dynamics": hashlib.sha256((checkpoint_dir /
                        f"pooled_{config['dynamics_seed']}_best.pt").read_bytes()).hexdigest(),
                    "reward": hashlib.sha256((checkpoint_dir /
                        f"reward_{config['reward_seed']}_best.pt").read_bytes()).hexdigest()},
                "config": config, "splits": metadata}
    (output / "coverage_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
