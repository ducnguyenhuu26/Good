"""Independent one-world CLSM future-law diagnostic with matched oracle probes.

This executable path checks the horizon and frozen-current comparison. It is
not the five-seed/twenty-world confirmatory C1/C2 experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from rwt.clsm.belief import LawFilter
from rwt.clsm.factory import build_clsm
from rwt.clsm.forecast import LawForecaster
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import verify_checkpoint_source
from rwt.clsm.scoring import energy_score
from experiments.rwt.clsm.build_datasets import _sha256
from experiments.rwt.clsm.collectors import SmoothActionCollector
from experiments.rwt.clsm.oracle_probes import fixed_probe_bank, oracle_signature
from experiments.rwt.clsm.physics_wrappers import HiddenPhysicsWorld
from experiments.rwt.clsm.world_schedule import make_schedule


def evaluate(env_id: str, track: str, checkpoint: Path, output_dir: Path,
             seed: int, probe_count: int | None = None) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("kind") != "clsm_approximate_gem":
        raise ValueError("checkpoint must contain fitted law, evolution and hierarchy")
    verify_checkpoint_source(payload)
    config = payload["config"]
    if config["track"] != track:
        raise ValueError("checkpoint and evaluation track disagree")
    schema = mujoco_nominal_schema(env_id)
    parts = build_clsm(schema, config)
    parts.law.load_state_dict(payload["law"])
    parts.evolution.load_state_dict(payload["evolution"])
    parts.hierarchy.load_state_dict(payload["hierarchy"])
    parts.law.eval(); parts.evolution.eval(); parts.hierarchy.eval()
    horizons = [0] + config["forecast"]["horizons"]
    schedule = make_schedule(track, "test", max(horizons) + 1, seed)
    world = HiddenPhysicsWorld(env_id, schedule, schema)
    try:
        obs, _ = world.reset(seed=seed + 100000)
        collector = SmoothActionCollector(schema.action_low, schema.action_high, seed + 200000)
        action = collector.act(obs)
        next_obs, _, _, _, _ = world.step(action)
    finally:
        world.close()
    filt = LawFilter(parts.law, parts.evolution, parts.hierarchy.new_memory(),
                     config["filter"]["components"], config["filter"]["max_dropped_mass"],
                     config["filter"]["covariance_jitter_initial"],
                     config["filter"]["covariance_jitter_max"])
    filt.observe(torch.as_tensor(obs), torch.as_tensor(action), torch.as_tensor(next_obs))
    public_belief = filt.components
    count = config["forecast"]["signature_probes"] if probe_count is None else probe_count
    bank, bank_path = fixed_probe_bank(env_id, count, output_dir.parent / "clsm_evaluator")
    evaluator_dir = output_dir.parent / "clsm_evaluator" / track / env_id / f"forecast_{seed}"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    public_probes = (torch.as_tensor(bank.obs, dtype=torch.float64),
                     torch.as_tensor(bank.action, dtype=torch.float64))
    forecaster = LawForecaster(parts.law, parts.evolution, filt.memory,
                              config["forecast"]["samples"])
    predicted = forecaster.sample_signatures(public_belief, public_probes, horizons,
                                              np.random.default_rng(seed + 400000))
    frozen = forecaster.sample_signatures(public_belief, public_probes, horizons,
                                           np.random.default_rng(seed + 400000), frozen=True)
    target = {h: oracle_signature(env_id, bank, schedule[h], parts.law.delta_scale)
              for h in horizons}
    np.savez_compressed(evaluator_dir / "oracle_signatures.npz",
                        **{f"h{h}": value.numpy() for h, value in target.items()})
    run_dir = checkpoint.parent / f"forecast_diagnostic_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "forecasts.npz",
                        **{f"clsm_h{h}": predicted[h].numpy() for h in horizons},
                        **{f"frozen_h{h}": frozen[h].numpy() for h in horizons})
    rows = []
    for horizon in horizons:
        clsmes = float(energy_score(predicted[horizon], target[horizon]))
        frozenes = float(energy_score(frozen[horizon], target[horizon]))
        rows.append({"world_seed": seed, "time": 0, "horizon": horizon,
                     "clsm_energy_score": clsmes, "frozen_energy_score": frozenes,
                     "skill": frozenes - clsmes})
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {"status": "single_world_diagnostic_only", "env": env_id, "track": track,
              "world_seed": seed, "checkpoint_sha256": _sha256(checkpoint),
              "public_probe_sha256": _sha256(bank_path),
              "probes": count, "samples": forecaster.samples,
              "horizons": horizons, "scores": rows,
              "oracle_artifact_dir": str(evaluator_dir),
              "confirmatory_c2": False}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--track", required=True, choices=("comp", "evol", "mem"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--probes", type=int, help="diagnostic-only probe-count override")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"env": args.env, "track": args.track,
                          "checkpoint": str(args.checkpoint), "seed": args.seed}))
        return
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint missing: {args.checkpoint}")
    result = evaluate(args.env, args.track, args.checkpoint, args.output_dir,
                      args.seed, args.probes)
    print(json.dumps({"status": result["status"], "scores": result["scores"]}))


if __name__ == "__main__":
    main()
