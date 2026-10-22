"""Evaluator-only oracle check of visit-specific information in MEM schedules.

It compares a previous-entry cache with the known population mean for matched
independent/correlated schedule conditions. No CLSM model is trained or scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from experiments.rwt.clsm.world_schedule import make_schedule


def _entry_pairs(schedule) -> np.ndarray:
    prior = {}
    pairs = []
    for time, law in enumerate(schedule):
        if time and law.code != schedule[time - 1].code and law.code:
            active = next(j for j in range(3) if (law.code >> j) & 1)
            value = (law.motor_1, law.motor_2, law.damping)[active]
            if law.code in prior:
                pairs.append((active, prior[law.code], value))
            prior[law.code] = value
    return np.asarray(pairs, dtype=np.float64).reshape(-1, 3)


def audit(first_seed: int = 100, worlds: int = 50, length: int = 500,
          bootstrap_draws: int = 2000) -> dict:
    if worlds < 2 or length < 2 or bootstrap_draws < 1:
        raise ValueError("need multiple independent worlds and positive length/draws")
    conditions = ("mem_independent", "mem_correlated")
    collected = {condition: [] for condition in conditions}
    per_world = {condition: [] for condition in conditions}
    switch_count = {condition: 0 for condition in conditions}
    for seed in range(first_seed, first_seed + worlds):
        schedules = {condition: make_schedule("mem", "test", length, seed, condition)
                     for condition in conditions}
        if [(law.code, law.reset) for law in schedules[conditions[0]]] != [
                (law.code, law.reset) for law in schedules[conditions[1]]]:
            raise AssertionError("MEM controls have different structural schedules")
        for condition, schedule in schedules.items():
            switch_count[condition] += sum(schedule[t].code != schedule[t - 1].code
                                           for t in range(1, length))
            pairs = _entry_pairs(schedule)
            if not len(pairs):
                continue
            active = pairs[:, 0].astype(int)
            center = np.where(active == 2, 1.65, .675)
            width = np.where(active == 2, .7, .35)
            old = (pairs[:, 1] - center) / width
            current = (pairs[:, 2] - center) / width
            collected[condition].extend(zip(old.tolist(), current.tolist()))
            per_world[condition].append(float(np.mean(current ** 2 - (current - old) ** 2)))
    rng = np.random.default_rng(3901)
    results = {}
    for condition in conditions:
        pairs = np.asarray(collected[condition], dtype=np.float64)
        gains = np.asarray(per_world[condition], dtype=np.float64)
        if len(pairs) < 2 or len(gains) < 2:
            raise RuntimeError("not enough recurrent visits for oracle diagnostic")
        old, current = pairs[:, 0], pairs[:, 1]
        draws = rng.choice(gains, size=(bootstrap_draws, len(gains)), replace=True).mean(1)
        results[condition] = {
            "recurrent_entry_pairs": len(pairs),
            "worlds_with_pairs": len(gains),
            "structural_switches": switch_count[condition],
            "normalized_entry_correlation": float(np.corrcoef(old, current)[0, 1]),
            "previous_value_mse": float(np.mean((current - old) ** 2)),
            "population_mean_mse": float(np.mean(current ** 2)),
            "mean_world_gain_population_minus_cache": float(gains.mean()),
            "world_bootstrap_95pct_ci": np.quantile(draws, [.025, .975]).tolist(),
        }
    return {"status": "oracle_generator_diagnostic_only",
            "first_seed": first_seed, "independent_worlds": worlds,
            "world_length": length, "bootstrap_draws": bootstrap_draws,
            "matched_structural_schedules": True,
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--worlds", type=int, default=50)
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.first_seed, args.worlds, args.length)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
