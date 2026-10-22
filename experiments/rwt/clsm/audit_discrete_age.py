"""Exact no-emission semi-Markov age reference versus support-preserving pruning.

This is an evaluator diagnostic. It does not replace the continuous CLSM
filter or establish accuracy under physical observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from rwt.clsm.evolution import LawEvolution


State = tuple[int, int, int]  # code, exact age, scratch/reuse hypothesis


def _step(distribution: dict[State, float], transition: np.ndarray,
          hazard: np.ndarray) -> dict[State, float]:
    predicted: dict[State, float] = {}
    for (code, age, hypothesis), weight in distribution.items():
        switch = weight * float(hazard[age])
        stay_key = (code, age + 1, hypothesis)
        predicted[stay_key] = predicted.get(stay_key, 0.) + weight - switch
        for target in range(8):
            if target != code:
                key = (target, 0, hypothesis)
                predicted[key] = predicted.get(key, 0.) + switch * float(transition[code, target])
    return predicted


def _compress(distribution: dict[State, float], budget: int) -> tuple[dict[State, float], float]:
    ordered = sorted(distribution, key=lambda key: distribution[key], reverse=True)
    required: dict[tuple[int, int], State] = {}
    for key in ordered:
        required.setdefault((key[0], key[2]), key)
    if len(required) > budget:
        raise ValueError("budget cannot preserve code/hypothesis support")
    retained = list(required.values())
    seen = set(retained)
    for key in ordered:
        if len(retained) >= budget:
            break
        if key not in seen:
            retained.append(key)
            seen.add(key)
    kept = sum(distribution[key] for key in retained)
    return {key: distribution[key] / kept for key in retained}, max(0., 1. - kept)


def audit(horizon: int, budgets: tuple[int, ...], scale: float = 20.,
          shape: float = 1.4) -> dict:
    if horizon < 1 or min(budgets) < 16 or scale <= 0 or shape <= 0:
        raise ValueError("invalid age audit configuration")
    ages = np.arange(horizon + 1, dtype=np.float64)
    hazard = -np.expm1(-(((ages + 1) / scale) ** shape - (ages / scale) ** shape))
    transition = LawEvolution().structural_log_transition().exp().detach().numpy()
    initial = {(code, 0, hypothesis): 1. / 16
               for code in range(8) for hypothesis in (0, 1)}
    exact = initial.copy()
    approximate = {budget: initial.copy() for budget in budgets}
    max_drop = {budget: 0. for budget in budgets}
    violation_steps = {budget: 0 for budget in budgets}
    last_drop = {budget: 0. for budget in budgets}
    for _ in range(horizon):
        exact = _step(exact, transition, hazard)
        for budget in budgets:
            approximate[budget], dropped = _compress(
                _step(approximate[budget], transition, hazard), budget)
            max_drop[budget] = max(max_drop[budget], dropped)
            violation_steps[budget] += dropped > .001
            last_drop[budget] = dropped
    if abs(sum(exact.values()) - 1.) > 1e-10:
        raise AssertionError("exact age recursion lost probability mass")
    rows = []
    for budget in budgets:
        current = approximate[budget]
        support = exact.keys() | current.keys()
        tv = .5 * sum(abs(exact.get(key, 0.) - current.get(key, 0.)) for key in support)
        rows.append({"component_budget": budget,
                     "max_step_dropped_mass": max_drop[budget],
                     "final_step_dropped_mass": last_drop[budget],
                     "steps_above_0_1pct_gate": violation_steps[budget],
                     "tv_to_exact_at_horizon": tv,
                     "retained_support": len(current)})
    return {"condition": "no_emission_discrete_age_reference",
            "horizon": horizon, "weibull_scale": scale, "weibull_shape": shape,
            "exact_support": len(exact), "results": rows,
            "interpretation": "numerical support diagnostic, not physical-world validation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--scale", type=float, default=20.)
    parser.add_argument("--shape", type=float, default=1.4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.horizon, tuple(args.budgets), args.scale, args.shape)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
