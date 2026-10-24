"""Experiment A: old-physics identifiability versus bounded estimation from 96 old transitions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q1.diagnostics.feasibility_abc import (
    estimate_old_params, functional_class_ids, parameter_vector, params_from_vector,
    score_candidates, vector_response,
)
from q1.simulator import PROGRAMS, apply_program
from scripts.q1_feasibility_common import (
    get_stage_data, load_protocol, params_for, probe_rows, sha256_file,
    sha256_tree, write_json,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _program_label(index: int) -> str:
    return "identity" if not PROGRAMS[index] else "-".join(f"U{x+1}" for x in PROGRAMS[index])


def _aggregate(rows: list[dict], z: float = 1.96) -> dict:
    out = {}
    for regime in ("IDLIKE_DEVELOPMENT", "DEVELOPMENT_OOD"):
        for method in ("known_old_physics", "estimated_old_physics"):
            for budget in (5, 10, 20, 50, 100):
                subset = [r for r in rows if r["regime"] == regime and r["method"] == method and r["budget"] == budget]
                by_bg = {}
                for row in subset:
                    key = row["environment_id"]
                    by_bg.setdefault(key, []).append(row)
                background_acc = [np.mean([x["functional_class_correct"] for x in v]) for v in by_bg.values()]
                identity_acc = [np.mean([x["program_identity_correct"] for x in v]) for v in by_bg.values()]
                margins = [np.mean([x["true_vs_best_wrong_class_margin_mse"] for x in v]) for v in by_bg.values()]
                mean = float(np.mean(background_acc))
                se = float(np.std(background_acc, ddof=1) / np.sqrt(len(background_acc))) if len(background_acc)>1 else 0.0
                out[f"{regime}/{method}/N{budget}"] = {
                    "n_backgrounds": len(by_bg), "functional_class_accuracy_mean": mean,
                    "functional_class_accuracy_ci95_background": [mean-z*se, mean+z*se],
                    "program_identity_accuracy_mean": float(np.mean(identity_acc)),
                    "mean_true_vs_best_wrong_class_margin_mse": float(np.mean(margins)),
                    "uncertainty_unit": "background after within-background program averaging"}
    return out


def run(config_path: str, out_path: str) -> dict:
    cfg, manifest, out = load_protocol(config_path, out_path, initialize=True)
    split = json.loads((out / "original_split_manifest.json").read_text(encoding="utf-8"))
    split_cfg = yaml.safe_load((ROOT / "configs/q1_splits.yaml").read_text(encoding="utf-8"))
    dt, sigma = float(cfg["dt"]), float(cfg["noise_sigma"])
    budgets = [int(x) for x in cfg["budgets"]]
    n_adapt, n_query = int(cfg["adaptation_trajectory_size"]), int(cfg["assessment_query_size"])
    start = time.perf_counter()
    protected = [ROOT / "results/q1/u4_root_cause_v1", ROOT / "results/q1/repair_v1_m3_effect_ablation"]
    protected_hashes = {str(p.relative_to(ROOT)): sha256_tree(p) for p in protected if p.exists()}
    all_rows, estimate_rows, confusion_rows = [], [], []
    regimes = ("IDLIKE_DEVELOPMENT", "DEVELOPMENT_OOD")
    for regime_idx, regime in enumerate(regimes):
        count = int(cfg["idlike_backgrounds"] if regime.startswith("IDLIKE") else cfg["development_ood_backgrounds"])
        for env in range(count):
            params = params_for(manifest, regime, env, split, split_cfg["parameter_ranges"])
            seed = int(cfg["seed"]) + regime_idx * 4_000_000 + env * 100_003
            old, _, _ = get_stage_data(params, (), seed=seed, n_support=int(cfg["old_context_size"]),
                n_adapt=n_adapt, n_query=n_query, dt=dt, sigma=sigma)
            old_states = old.states[:-1]
            old_actions = old.actions
            old_dv = old.delta_v
            theta_hat, fit_meta = estimate_old_params(old_states, old_actions, old_dv, dt=dt,
                seed=seed + 700001, max_nfev=int(cfg["parameter_fit"]["max_nfev"]),
                initializations=int(cfg["parameter_fit"]["multistarts"]))
            classes, _ = functional_class_ids(params, dt=dt, seed=seed + 77001)
            err = parameter_vector(theta_hat) - parameter_vector(params)  # reporting only, after fit frozen
            estimate_rows.append({"regime": regime, "background": env,
                "environment_id": (manifest["idlike"]["environment_ids"][env] if regime.startswith("IDLIKE")
                                   else manifest["development_ood"]["environment_ids"][env]),
                "seed": seed, "theta_hat": parameter_vector(theta_hat).tolist(),
                "theta_true_reporting_only": parameter_vector(params).tolist(),
                "parameter_error_reporting_only": err.tolist(), **fit_meta})
            for true_idx, true_program in enumerate(PROGRAMS):
                old_same, adaptation, query = get_stage_data(params, true_program, seed=seed,
                    n_support=int(cfg["old_context_size"]), n_adapt=n_adapt, n_query=n_query,
                    dt=dt, sigma=sigma)
                if not np.array_equal(old_same.states, old.states) or not np.array_equal(old_same.actions, old.actions):
                    raise AssertionError("old support must be identical across programs for a background")
                query_x = probe_rows(query)[:n_query]
                true_query_new = vector_response(apply_program(params, true_program), query_x, dt=dt)
                for budget in budgets:
                    probes = np.c_[adaptation.states[:budget], adaptation.actions[:budget]]
                    observed = adaptation.delta_v[:budget]
                    for method, theta in (("known_old_physics", params), ("estimated_old_physics", theta_hat)):
                        scores = score_candidates(theta, probes, observed, dt=dt)
                        order = np.argsort(scores, kind="stable")
                        pred_idx = int(order[0])
                        true_class = classes[true_idx]
                        pred_class = classes[pred_idx]
                        wrong_classes = [j for j in order if classes[int(j)] != true_class]
                        best_wrong_idx = int(wrong_classes[0]) if wrong_classes else pred_idx
                        candidate_after = vector_response(apply_program(theta, PROGRAMS[pred_idx]), query_x, dt=dt)
                        old_law_est = vector_response(theta, query_x, dt=dt)
                        identity_after = old_law_est
                        true_old_query = vector_response(params, query_x, dt=dt)
                        score_true = float(scores[true_idx])
                        row = {"regime": regime, "environment_id": estimate_rows[-1]["environment_id"],
                            "background": env, "seed": seed, "true_program_index": true_idx,
                            "true_program": json.dumps(true_program), "true_class": true_class,
                            "budget": budget, "method": method, "predicted_program_index": pred_idx,
                            "predicted_program": json.dumps(PROGRAMS[pred_idx]),
                            "predicted_class": pred_class,
                            "program_identity_correct": int(pred_idx == true_idx),
                            "functional_class_correct": int(pred_class == true_class),
                            "true_program_score_mse": score_true,
                            "best_score_mse": float(scores[pred_idx]),
                            "best_second_margin_mse": float(scores[order[1]] - scores[order[0]]) if len(order)>1 else None,
                            "true_vs_best_wrong_class_margin_mse": float(scores[best_wrong_idx] - score_true),
                            "query_true_candidate_conditional_mse": float(np.mean((vector_response(
                                apply_program(theta, true_program), query_x, dt=dt) - true_query_new) ** 2)),
                            "query_selected_candidate_conditional_mse": float(np.mean((candidate_after-true_query_new)**2)),
                            "query_no_change_baseline_conditional_mse": float(np.mean((identity_after-true_query_new)**2)),
                            "query_true_old_law_mse": float(np.mean((old_law_est-true_old_query)**2)),
                            "candidate_no_change_score_mse": float(scores[PROGRAMS.index(())]),
                            "functional_classes_are_oracle_diagnostic": True}
                        all_rows.append(row)
            # Focused confusion for actual U4 and identity cases, retaining other predictions.
            for budget in budgets:
                for method in ("known_old_physics", "estimated_old_physics"):
                    for true_idx in (PROGRAMS.index(()), PROGRAMS.index((3,))):
                        subset = [r for r in all_rows if r["regime"] == regime and r["background"] == env
                                  and r["budget"] == budget and r["method"] == method
                                  and r["true_program_index"] == true_idx]
                        label = "U4" if true_idx == PROGRAMS.index((3,)) else "identity"
                        confusion_rows.append({"regime": regime, "background": env, "environment_id": estimate_rows[-1]["environment_id"],
                            "budget": budget, "method": method, "true_case": label,
                            "pred_identity": int(any(r["predicted_program_index"] == PROGRAMS.index(()) for r in subset)),
                            "pred_U4": int(any(r["predicted_program_index"] == PROGRAMS.index((3,)) for r in subset)),
                            "pred_other": int(not any(r["predicted_program_index"] in (PROGRAMS.index(()), PROGRAMS.index((3,))) for r in subset))})
    a_metrics = {"status": "DEVELOPMENT_DIAGNOSTIC", "programs": [list(p) for p in PROGRAMS],
        "functional_class_accuracy_is_distinct_from_program_identity": True,
        "known_old_physics_is_privileged": True, "estimated_old_physics_is_equation_privileged": True,
        "metrics": _aggregate(all_rows), "u4_identity_confusion": {}}
    for regime in regimes:
        for method in ("known_old_physics", "estimated_old_physics"):
            for budget in budgets:
                sub = [r for r in confusion_rows if r["regime"] == regime and r["method"] == method and r["budget"] == budget]
                a_metrics["u4_identity_confusion"][f"{regime}/{method}/N{budget}"] = {
                    key: sum(r[key] for r in sub if r["true_case"] == true_case)
                    for true_case in ("identity", "U4")
                    for key in ("pred_identity", "pred_U4", "pred_other")}
    legacy_after = {str(p.relative_to(ROOT)): sha256_tree(p) for p in protected if p.exists()}
    if protected_hashes != legacy_after:
        raise RuntimeError("A changed a protected legacy result tree")
    write_json(out / "a_system_id_metrics.json", a_metrics)
    write_json(out / "a_parameter_estimates.json", {"backgrounds": estimate_rows,
        "scope": "theta_hat fitted from 96 old-stage noisy transitions; theta_true fields are reporting-only"})
    write_json(out / "a_u4_identity_confusion.json", a_metrics["u4_identity_confusion"])
    _write_csv(out / "a_per_background.csv", all_rows)
    (out / "a_status.json").write_text(json.dumps({"status": "COMPLETE", "wall_seconds": time.perf_counter()-start,
        "protected_tree_hashes_before": protected_hashes, "protected_tree_hashes_after": legacy_after}, indent=2), encoding="utf-8")
    return {"rows": len(all_rows), "background_parameter_fits": len(estimate_rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/q1_feasibility_abc.yaml")
    ap.add_argument("--out", default="results/q1/feasibility_abc_v1")
    args = ap.parse_args()
    try:
        result = run(args.config, args.out)
    except Exception as exc:
        print(f"Experiment A failed: {type(exc).__name__}: {exc}")
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()
