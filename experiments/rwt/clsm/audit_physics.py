"""Matched-state MuJoCo intervention audit; evaluator-only oracle probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Direct `python experiments/rwt/clsm/audit_physics.py` keeps the script's
# directory on sys.path, so add the repository root for evaluator imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mujoco
import numpy as np

from rwt.clsm.config import load_config
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld, stationary_public_reward


def _snapshot(model, data):
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, spec)
    return state, data.qacc_warmstart.copy()


def _restore_snapshot(model, data, snapshot):
    state, warmstart = snapshot
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.qacc_warmstart[:] = warmstart
    mujoco.mj_forward(model, data)


def audit(env_id: str, seed: int = 41, probes: int = 8) -> dict:
    laws = (HiddenLaw(0), HiddenLaw(1, motor_1=.7), HiddenLaw(2, motor_2=.7),
            HiddenLaw(4, damping=1.5), HiddenLaw(3, motor_1=.7, motor_2=.7),
            HiddenLaw(5, motor_1=.7, damping=1.5))
    env = HiddenPhysicsWorld(env_id, [HiddenLaw(0)] * max(probes + 1, 16))
    rng = np.random.default_rng(seed)
    model, data = env.unwrapped.model, env.unwrapped.data
    baseline_gain, baseline_damping = env._base_gain.copy(), env._base_damping.copy()
    separations = []
    rewards_invariant = True
    repeat_error = 0.
    matched_obs_error = 0.
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(probes):
            snapshot = _snapshot(model, data)
            action = rng.uniform(env.action_space.low, env.action_space.high) * .5
            predictions = []
            reference_obs = None
            reference_reward = None
            for law in laws:
                env._apply(law)
                _restore_snapshot(model, data, snapshot)
                prior_obs = env.unwrapped._get_obs().copy()
                next_obs = env.env.step(action)[0]
                predictions.append(next_obs.copy())
                if reference_obs is None:
                    reference_obs = prior_obs
                    reference_reward = stationary_public_reward(prior_obs, action)
                matched_obs_error = max(matched_obs_error, float(np.max(np.abs(prior_obs - reference_obs))))
                rewards_invariant &= bool(np.isclose(stationary_public_reward(prior_obs, action),
                                                      reference_reward, rtol=0, atol=1e-12))
            env._apply(laws[0])
            _restore_snapshot(model, data, snapshot)
            first = env.env.step(action)[0]
            env._apply(laws[0])
            _restore_snapshot(model, data, snapshot)
            second = env.env.step(action)[0]
            repeat_error = max(repeat_error, float(np.max(np.abs(first - second))))
            separations.append({f"0_vs_{i}": float(np.linalg.norm(predictions[0] - predictions[i]))
                                for i in range(1, len(laws))})
            obs = second
        env._restore()
        restoration = bool(np.array_equal(model.actuator_gainprm, baseline_gain)
                           and np.array_equal(model.dof_damping, baseline_damping))
        return {"environment": env_id, "seed": seed, "probes": probes,
                "law_codes": [law.code for law in laws],
                "paired_separations": separations,
                "repeat_max_abs_error": repeat_error,
                "matched_public_obs_max_abs_error": matched_obs_error,
                "stationary_reward_invariant": bool(rewards_invariant),
                "physics_restored": restoration,
                "nominal_damping": baseline_damping.tolist(),
                "nominal_gainprm": baseline_gain.tolist()}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"env": args.env, "config_track": config["track"], "probes": args.probes}))
        return
    try:
        report = audit(args.env, args.seed, args.probes)
    except ValueError as error:
        raise SystemExit(f"physical adapter audit failed: {error}") from error
    path = args.output_dir / "physics_audits" / f"{args.env}_{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "repeat_max_abs_error": report["repeat_max_abs_error"],
                      "physics_restored": report["physics_restored"]}))


if __name__ == "__main__":
    main()
