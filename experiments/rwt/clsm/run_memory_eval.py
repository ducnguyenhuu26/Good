"""One-world matched-data scratch/reuse memory diagnostic.

The result is a projected-law scoring proxy, not confirmatory C3 evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from rwt.clsm.belief import LawFilter
from rwt.clsm.factory import build_clsm
from rwt.clsm.forecast import LawForecaster
from rwt.clsm.memory import TheoryMemory
from rwt.clsm.schemas import mujoco_nominal_schema
from rwt.clsm.serialization import assert_deployable_checkpoint, verify_checkpoint_source
from rwt.clsm.scoring import energy_score
from experiments.rwt.clsm.build_datasets import _sha256
from experiments.rwt.clsm.collectors import SmoothActionCollector
from experiments.rwt.clsm.oracle_probes import fixed_probe_bank, oracle_signature
from experiments.rwt.clsm.physics_wrappers import HiddenPhysicsWorld
from experiments.rwt.clsm.world_schedule import make_schedule


def evaluate(env_id: str, checkpoint: Path, output_dir: Path, seed: int,
             budgets: list[int], probes: int | None = None,
             component_budget: int | None = None) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert_deployable_checkpoint(payload)
    if payload.get("kind") != "clsm_approximate_gem" or payload["config"]["track"] != "mem":
        raise ValueError("memory evaluation requires a MEM approximate-GEM checkpoint")
    verify_checkpoint_source(payload)
    if not budgets or min(budgets) < 1 or len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be distinct positive transition counts")
    config = payload["config"]
    filter_budget = config["filter"]["components"] if component_budget is None else int(component_budget)
    if filter_budget < 16:
        raise ValueError("component budget must preserve all code/model hypotheses")
    schema = mujoco_nominal_schema(env_id)
    parts = build_clsm(schema, config)
    parts.law.load_state_dict(payload["law"])
    parts.evolution.load_state_dict(payload["evolution"])
    parts.hierarchy.load_state_dict(payload["hierarchy"])
    parts.law.eval(); parts.evolution.eval(); parts.hierarchy.eval()
    condition = config.get("schedule_condition", "in_family")
    schedule = make_schedule("mem", "test", max(budgets), seed, condition=condition)
    world = HiddenPhysicsWorld(env_id, schedule, schema)
    transitions = []
    try:
        obs, _ = world.reset(seed=seed + 100000)
        collector = SmoothActionCollector(schema.action_low, schema.action_high, seed + 200000)
        for t in range(max(budgets)):
            action = collector.act(obs)
            next_obs, _, terminated, _, _ = world.step(action)
            transitions.append((torch.as_tensor(obs.copy()), torch.as_tensor(action.copy()),
                                torch.as_tensor(next_obs.copy())))
            obs = next_obs
            if terminated and t + 1 < max(budgets):
                raise RuntimeError("physical episode terminated before requested adaptation budget")
    finally:
        world.close()
    full_memory = parts.hierarchy.new_memory()
    scratch_memory = TheoryMemory(float(parts.hierarchy.prior_mean.detach()),
                                  float(parts.hierarchy.prior_var.detach().sqrt()),
                                  float(parts.hierarchy.interaction_var.detach()),
                                  float(parts.hierarchy.recurrence_var.detach()), 1.)
    filters = {
        "reuse_mixture": LawFilter(parts.law, parts.evolution, full_memory,
                                    filter_budget, config["filter"]["max_dropped_mass"],
                                    config["filter"]["covariance_jitter_initial"],
                                    config["filter"]["covariance_jitter_max"]),
        "scratch_only": LawFilter(parts.law, parts.evolution, scratch_memory,
                                   filter_budget, config["filter"]["max_dropped_mass"],
                                   config["filter"]["covariance_jitter_initial"],
                                   config["filter"]["covariance_jitter_max"]),
    }
    count = config["forecast"]["signature_probes"] if probes is None else probes
    bank, bank_path = fixed_probe_bank(env_id, count, output_dir.parent / "clsm_evaluator")
    evaluator_dir = output_dir.parent / "clsm_evaluator" / "mem" / env_id / f"memory_{condition}_{seed}"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    public_probes = (torch.as_tensor(bank.obs, dtype=torch.float64),
                     torch.as_tensor(bank.action, dtype=torch.float64))
    rows = []
    predictions = {}
    targets = {}
    elapsed = {name: 0. for name in filters}
    worst_drop = {name: 0. for name in filters}
    for t, (obs, action, next_obs) in enumerate(transitions):
        for name, filt in filters.items():
            start = perf_counter()
            filt.observe(obs, action, next_obs)
            elapsed[name] += perf_counter() - start
        budget = t + 1
        if budget in budgets:
            target = oracle_signature(env_id, bank, schedule[t], parts.law.delta_scale)
            targets[f"n{budget}"] = target.numpy()
            scores = {}
            for name, filt in filters.items():
                start = perf_counter()
                draws = LawForecaster(parts.law, parts.evolution, filt.memory,
                                     config["forecast"]["samples"]).sample_signatures(
                                         filt.components, public_probes, [0],
                                         np.random.default_rng(seed + 400000 + budget))[0]
                elapsed[name] += perf_counter() - start
                predictions[f"{name}_n{budget}"] = draws.numpy()
                scores[name] = float(energy_score(draws, target))
            rows.append({"world_seed": seed, "budget": budget,
                         "reuse_energy_score": scores["reuse_mixture"],
                         "scratch_energy_score": scores["scratch_only"],
                         "gain": scores["scratch_only"] - scores["reuse_mixture"],
                         "reuse_components": len(filters["reuse_mixture"].components),
                         "scratch_components": len(filters["scratch_only"].components),
                         "reuse_max_dropped_mass": worst_drop["reuse_mixture"],
                         "scratch_max_dropped_mass": worst_drop["scratch_only"]})
        if t + 1 < len(transitions):
            for name, filt in filters.items():
                start = perf_counter()
                step = filt.evolve()
                elapsed[name] += perf_counter() - start
                worst_drop[name] = max(worst_drop[name], step.dropped_mass)
    np.savez_compressed(evaluator_dir / "oracle_signatures.npz", **targets)
    suffix = "" if filter_budget == config["filter"]["components"] else f"_components_{filter_budget}"
    if condition != "in_family":
        suffix = f"_{condition}{suffix}"
    run_dir = checkpoint.parent / f"memory_diagnostic_{seed}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "predictions.npz", **predictions)
    with (run_dir / "memory_adaptation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {"status": "single_world_memory_diagnostic_only", "budgets": budgets,
              "schedule_condition": condition,
              "component_budget": filter_budget,
              "component_budget_ablation": filter_budget != config["filter"]["components"],
              "scores": rows, "seconds": elapsed, "max_dropped_mass": worst_drop,
              "inference_degraded": any(value > config["filter"]["max_dropped_mass"] for value in worst_drop.values()),
              "checkpoint_sha256": _sha256(checkpoint), "probe_sha256": _sha256(bank_path),
              "matched_compute_scratch": False, "confirmatory_c3": False,
              "oracle_artifact_dir": str(evaluator_dir)}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--track", required=True, choices=("mem",))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rwt/clsm"))
    parser.add_argument("--seed", type=int, default=801)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--probes", type=int, help="diagnostic-only probe-count override")
    parser.add_argument("--component-budget", type=int,
                        help="separately reported accuracy/computation ablation; default is locked config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"env": args.env, "checkpoint": str(args.checkpoint),
                          "budgets": args.budgets}))
        return
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint missing: {args.checkpoint}")
    result = evaluate(args.env, args.checkpoint, args.output_dir, args.seed,
                      args.budgets, args.probes, args.component_budget)
    print(json.dumps({"status": result["status"], "scores": result["scores"],
                      "inference_degraded": result["inference_degraded"]}))


if __name__ == "__main__":
    main()
