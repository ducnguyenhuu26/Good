"""Matched A/B/D physics audit, then held-out C evaluation of a frozen proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from rwt.stage3a import DEFAULT_REGIMES, PhysicsRegime, RecurrentHalfCheetah


def snapshots(seed: int = 920) -> list[tuple]:
    rng = np.random.default_rng(seed)
    env = RecurrentHalfCheetah([("A", 40)])
    result, previous = [], np.zeros(6)
    try:
        env.reset(seed=seed)
        for index in range(32):
            action = (np.clip(.55 * previous + .45 * rng.normal(0, .8, 6), -1, 1)
                      if index < 16 else rng.uniform(-1, 1, 6))
            qpos, qvel = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
            if index >= 16:
                qvel *= 1.8
            result.append(("train_like" if index < 16 else "heldout_state_action",
                           qpos, qvel, action.copy()))
            env.step(action)
            previous = action
    finally:
        env.close()
    return result


def transition(regime: str, regimes: dict, qpos, qvel, action) -> np.ndarray:
    env = RecurrentHalfCheetah([(regime, 1)], regimes)
    try:
        env.reset(seed=920)
        env.unwrapped.set_state(qpos, qvel)
        result, _, _, _, _ = env.step(action)
        return result
    finally:
        env.close()


def abc_separation(regimes: dict, probes: list[tuple]) -> dict:
    values = {}
    for region in ("train_like", "heldout_state_action"):
        distances = []
        for location, qpos, qvel, action in probes:
            if location != region:
                continue
            outputs = {r: transition(r, regimes, qpos, qvel, action) for r in ("A", "B", "D")}
            distances.append((np.linalg.norm(outputs["A"] - outputs["B"]),
                              np.linalg.norm(outputs["A"] - outputs["D"])))
        means = np.mean(distances, axis=0)
        values[region] = {"A_B_l2": float(means[0]), "A_D_l2": float(means[1])}
    return values


def stability(regimes: dict, names: tuple[str, ...]) -> dict:
    results = {}
    for policy in ("smooth_random", "bounded_oscillator"):
        for regime in names:
            speeds, rewards, max_abs, finite = [], [], 0.0, True
            for seed in (101, 102, 103):
                env = RecurrentHalfCheetah([(regime, 200)], regimes)
                rng, previous = np.random.default_rng(seed), np.zeros(6)
                try:
                    state, _ = env.reset(seed=seed)
                    for step in range(200):
                        action = (np.clip(.55 * previous + .45 * rng.normal(0, .8, 6), -1, 1)
                                  if policy == "smooth_random" else
                                  .8 * np.sin(.3 * step + np.arange(6) * np.pi / 3))
                        state, reward, _, _, _ = env.step(action)
                        finite = finite and bool(np.isfinite(state).all() and np.isfinite(reward))
                        max_abs = max(max_abs, float(np.abs(state).max()))
                        speeds.append(float(state[8]))
                        rewards.append(float(reward))
                        previous = action
                finally:
                    env.close()
            results[f"{policy}_{regime}"] = {"finite": finite, "max_abs_observation": max_abs,
                                              "mean_x_velocity": float(np.mean(speeds)),
                                              "mean_reward": float(np.mean(rewards))}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path,
                        default=Path("experiments/rwt/configs/stage3b/proposed_regimes.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/rwt/stage3b_proposed_physics.json"))
    args = parser.parse_args()
    config = yaml.safe_load(args.proposal.read_text(encoding="utf-8"))
    if set(config) != {"A", "B", "D", "C"}:
        raise ValueError("proposal must contain A/B/D/C")
    proposed = {key: PhysicsRegime(float(value["friction"]), float(value["actuator_gain"]))
                for key, value in config.items()}
    if proposed["A"] != DEFAULT_REGIMES["A"]:
        raise ValueError("proposal must retain A reference physics")
    probes = snapshots()
    current_abd = abc_separation(DEFAULT_REGIMES, probes)
    proposed_abd = abc_separation(proposed, probes)
    proposed_abd_stability = stability(proposed, ("A", "B", "D"))
    if not all(item["finite"] and item["max_abs_observation"] < 100
               for item in proposed_abd_stability.values()):
        raise FloatingPointError("proposed A/B/D unstable on smoke trajectories")
    frozen_digest = hashlib.sha256(args.proposal.read_bytes()).hexdigest()
    # From here onward C is evaluation only; no proposal parameters change.
    c_stability = stability(proposed, ("C",))
    closure = []
    for _, qpos, qvel, action in probes:
        outputs = {r: transition(r, proposed, qpos, qvel, action) for r in ("A", "B", "D", "C")}
        residual = outputs["C"] - outputs["B"] - outputs["D"] + outputs["A"]
        denominator = (np.linalg.norm(outputs["B"] - outputs["A"])
                       + np.linalg.norm(outputs["D"] - outputs["A"]))
        closure.append((np.linalg.norm(residual), np.linalg.norm(residual) / max(denominator, 1e-12)))
    report = {"frozen_proposal_sha256": frozen_digest, "proposal": config,
              "current_abd_separation": current_abd,
              "proposed_abd_separation": proposed_abd,
              "proposed_abd_stability": proposed_abd_stability,
              "heldout_c_stability": c_stability,
              "heldout_c_closure_l2": float(np.mean(closure, axis=0)[0]),
              "heldout_c_closure_relative": float(np.mean(closure, axis=0)[1])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
