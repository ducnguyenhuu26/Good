"""Held-out composition probe diagnostic versus matched pooled public MLP."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from rwt.clsm.baselines import PooledGaussianMLP
from rwt.clsm.belief import LawFilter
from rwt.clsm.factory import build_clsm
from rwt.clsm.forecast import LawForecaster
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import verify_checkpoint_source
from rwt.clsm.scoring import energy_score
from experiments.rwt.clsm.build_datasets import _sha256
from experiments.rwt.clsm.collectors import SmoothActionCollector
from experiments.rwt.clsm.oracle_probes import fixed_probe_bank, oracle_signature
from experiments.rwt.clsm.physics_wrappers import HiddenLaw, HiddenPhysicsWorld


HELD_OUT_CODES = (3, 5, 6, 7)


def evaluate(env_id: str, checkpoint: Path, pooled_checkpoint: Path,
             output_dir: Path, seed: int, adaptation_steps: int = 4,
             probes: int | None = None) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    baseline_payload = torch.load(pooled_checkpoint, map_location="cpu", weights_only=True)
    if payload.get("kind") != "clsm_approximate_gem" or payload["config"]["track"] != "comp":
        raise ValueError("COMP approximate-GEM checkpoint required")
    if baseline_payload.get("kind") != "pooled_mlp":
        raise ValueError("matched pooled MLP checkpoint required")
    verify_checkpoint_source(payload)
    verify_checkpoint_source(baseline_payload)
    if payload["data_hashes"] != baseline_payload["data_hashes"]:
        raise ValueError("CLSM and pooled baseline used different public/reference datasets")
    config = payload["config"]
    schema = mujoco_nominal_schema(env_id)
    parts = build_clsm(schema, config)
    parts.law.load_state_dict(payload["law"])
    parts.evolution.load_state_dict(payload["evolution"])
    parts.hierarchy.load_state_dict(payload["hierarchy"])
    parts.law.eval(); parts.evolution.eval(); parts.hierarchy.eval()
    pooled = PooledGaussianMLP(schema.observation_dim, schema.action_dim,
                               config["network"]["hidden_dim"],
                               config["emission"]["variance_floor"],
                               config["emission"]["variance_ceiling"])
    pooled.load_state_dict(baseline_payload["model"])
    pooled.eval()
    if not (torch.equal(parts.law.delta_scale, pooled.delta_scale)
            and torch.equal(parts.law.delta_mean, pooled.delta_mean)):
        raise ValueError("CLSM and pooled baseline have different evaluator target normalization")
    count = config["forecast"]["signature_probes"] if probes is None else probes
    bank, bank_path = fixed_probe_bank(env_id, count, output_dir.parent / "clsm_evaluator")
    evaluator_dir = output_dir.parent / "clsm_evaluator" / "comp" / env_id / f"composition_{seed}"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    public_probes = (torch.as_tensor(bank.obs, dtype=torch.float64),
                     torch.as_tensor(bank.action, dtype=torch.float64))
    with torch.no_grad():
        pooled_mean = pooled(*public_probes)[0]
        pooled_signature = ((pooled_mean - public_probes[0]) / pooled.delta_scale).flatten()
    rows = []
    oracle_targets = {}
    predicted = {}
    for code in HELD_OUT_CODES:
        law = HiddenLaw(code, motor_1=.7, motor_2=.7, damping=1.5)
        world = HiddenPhysicsWorld(env_id, [law] * adaptation_steps, schema)
        transitions = []
        try:
            obs, _ = world.reset(seed=seed + 100000 + code)
            collector = SmoothActionCollector(schema.action_low, schema.action_high,
                                              seed + 200000 + code)
            for t in range(adaptation_steps):
                action = collector.act(obs)
                next_obs, _, terminated, _, _ = world.step(action)
                transitions.append((obs.copy(), action.copy(), next_obs.copy()))
                obs = next_obs
                if terminated and t + 1 < adaptation_steps:
                    raise RuntimeError("physical episode ended before adaptation budget")
        finally:
            world.close()
        filt = LawFilter(parts.law, parts.evolution, parts.hierarchy.new_memory(),
                         config["filter"]["components"], config["filter"]["max_dropped_mass"],
                         config["filter"]["covariance_jitter_initial"],
                         config["filter"]["covariance_jitter_max"])
        worst_drop = 0.
        for t, (obs, action, next_obs) in enumerate(transitions):
            filt.observe(torch.as_tensor(obs), torch.as_tensor(action), torch.as_tensor(next_obs))
            if t + 1 < len(transitions):
                worst_drop = max(worst_drop, filt.evolve().dropped_mass)
        draws = LawForecaster(parts.law, parts.evolution, filt.memory,
                             config["forecast"]["samples"]).sample_signatures(
                                 filt.components, public_probes, [0],
                                 np.random.default_rng(seed + 400000 + code))[0]
        target = oracle_signature(env_id, bank, law, parts.law.delta_scale)
        oracle_targets[f"code{code}"] = target.numpy()
        predicted[f"code{code}"] = draws.numpy()
        clsmes = float(energy_score(draws, target))
        # A deterministic point-law comparator has zero pairwise term.
        pooled_es = float(torch.linalg.vector_norm(pooled_signature - target))
        rows.append({"world_seed": seed + 100000 + code, "held_out_code": code,
                     "adaptation_steps": adaptation_steps, "clsm_energy_score": clsmes,
                     "pooled_energy_score": pooled_es, "pooled_minus_clsm": pooled_es - clsmes,
                     "max_dropped_mass": worst_drop})
    np.savez_compressed(evaluator_dir / "oracle_signatures.npz", **oracle_targets)
    run_dir = checkpoint.parent / f"composition_diagnostic_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "predictions.npz", **predicted,
                        pooled=pooled_signature.numpy())
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {"status": "single_seed_composition_diagnostic_only", "scores": rows,
              "checkpoint_sha256": _sha256(checkpoint),
              "pooled_checkpoint_sha256": _sha256(pooled_checkpoint),
              "probe_sha256": _sha256(bank_path), "confirmatory_c1": False,
              "oracle_artifact_dir": str(evaluator_dir)}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pooled-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--adaptation-steps", type=int, default=4)
    parser.add_argument("--probes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"env": args.env, "checkpoint": str(args.checkpoint),
                          "pooled_checkpoint": str(args.pooled_checkpoint)}))
        return
    result = evaluate(args.env, args.checkpoint, args.pooled_checkpoint,
                      args.output_dir, args.seed, args.adaptation_steps, args.probes)
    print(json.dumps({"status": result["status"], "scores": result["scores"]}))


if __name__ == "__main__":
    main()
