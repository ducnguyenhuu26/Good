"""Finalize frozen pre-change outputs with document-prespecified audits.

This is postprocessing only: it never fits a model or changes the evaluation data.
It verifies matched-order noninferiority, packages training records, and records
the provenance of these deterministic summaries.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def background_means(rows, field):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["background"], []).append(float(row[field]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_bootstrap(a, b, seed, reps=5000):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("paired bootstrap requires shared backgrounds")
    diffs = np.asarray([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(reps, len(diffs)), replace=True).mean(axis=1)
    return {"n_backgrounds": len(keys), "mean_difference": float(diffs.mean()),
            "ci95_background_bootstrap": [float(x) for x in np.quantile(means, [.025, .975])]}


def finalize(out: Path) -> None:
    config_path = out / "config.resolved.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["decision_thresholds"] = {
        "minimum_independent_backgrounds_per_regime": 32,
        "block_Ieff_paired_bootstrap_lower_bound_vs_each_control_gt": 0.0,
        "U4_skill_paired_bootstrap_lower_bound_vs_train_matched_population_gt": 0.0,
        "U4_skill_bootstrap_lower_bound_gt": 0.0,
        "U1_U3_noninferiority_margin_vs_sinusoidal": 0.02,
        "ordered_pair_class_accuracy_minimum": 0.80,
        "ordered_pair_class_noninferiority_margin_vs_sinusoidal": 0.02,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with (out / "per_background_per_program.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    comparisons = summary["comparisons"]
    for regime in ("IDLIKE", "BACKGROUND_OOD"):
        for order in ("[0, 1]", "[1, 0]"):
            pair_rows = [r for r in rows if r["regime"] == regime and
                         r["estimator"] == "V5_RECALIBRATED" and r["N"] == "100" and
                         r["true_program"] == order]
            block = background_means([r for r in pair_rows if r["policy"] == "BLOCK_24_ENERGY_MATCHED"],
                                     "functional_class_correct")
            sine = background_means([r for r in pair_rows if r["policy"] == "ORIGINAL_SINUSOIDAL"],
                                    "functional_class_correct")
            comparisons[f"{regime}|BLOCK24_minus_sin|V5_RECALIBRATED|order={order}"] = \
                paired_bootstrap(block, sine, 20261801 + (0 if regime == "IDLIKE" else 10) +
                                 (0 if order == "[0, 1]" else 1))
    lower_bounds = [v["ci95_background_bootstrap"][0] for k, v in comparisons.items()
                    if "|BLOCK24_minus_sin|V5_RECALIBRATED|order=" in k]
    order_ni = len(lower_bounds) == 4 and min(lower_bounds) >= -0.02
    decision = summary["decision"]
    u4_positive = all(summary["metrics"][
        f"{regime}|BLOCK_24_ENERGY_MATCHED|V5_RECALIBRATED|oracle_code_U4|ORACLE_CODE_LEARNED|N=100"
    ]["ci95_background"][0] > 0 for regime in ("IDLIKE", "BACKGROUND_OOD"))
    decision["gates"]["U4_effect_skill_positive_both_regimes"] = u4_positive
    decision["gates"]["ordered_pair_class_noninferior_to_sinusoidal_margin_0.02"] = order_ni
    enough_n = decision["gates"].get("at_least_32_independent_backgrounds_per_regime", False)
    scientific_gates = [value for key, value in decision["gates"].items()
                        if key != "at_least_32_independent_backgrounds_per_regime"]
    decision["verdict"] = ("INCONCLUSIVE" if not enough_n else
                           ("Q1_PRECHANGE_PASS" if all(scientific_gates) else "FAIL"))
    decision["gates"]["ordered_pair_class_noninferior_to_sinusoidal_margin_0.02"] = order_ni
    summary["decision"] = decision
    summary["comparisons"] = comparisons
    summary["postprocessing"] = {
        "purpose": "document-prespecified order noninferiority and deliverable completeness",
        "model_fitting_or_metric_selection": False,
        "paired_order_margin": 0.02,
        "script_sha256": sha(Path(__file__)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "baseline_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    def ci(value, interval):
        return f"{value:.4g} [{interval[0]:.4g}, {interval[1]:.4g}]"

    report = [
        "# Q1 pre-change excitation development decision", "",
        f"**Verdict: `{decision['verdict']}`.** This is a separate development protocol. "
        "Original fixed-policy and post-change active verdicts remain FAIL.", "",
        "The run used 32 independent ID-like and 32 OOD backgrounds, 40 training-only calibration backgrounds, "
        "all 21 programs, and N=5/10/20/50/100. Bootstrap intervals resample backgrounds.", "",
        "## Primary results at N=100", "",
        "| Regime | Block functional-class accuracy | U4 oracle-code skill | Train-matched population skill | Paired U4 difference | Block I_eff(log b) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for regime, label in (("IDLIKE", "ID-like"), ("BACKGROUND_OOD", "OOD")):
        pol = "BLOCK_24_ENERGY_MATCHED"
        acc = summary["metrics"][f"{regime}|V5_RECALIBRATED|{pol}|N=100|functional_class_accuracy"]
        u4 = summary["metrics"][f"{regime}|{pol}|V5_RECALIBRATED|oracle_code_U4|ORACLE_CODE_LEARNED|N=100"]
        pop = summary["metrics"][f"{regime}|{pol}|V5_RECALIBRATED|oracle_code_U4|MATCHED_TRAIN_POPULATION|N=100"]
        diff = comparisons[f"{regime}|V5_RECALIBRATED|U4_minus_matched_train_population"]
        fi = summary["metrics"][f"{regime}|{pol}|I_eff_pre_logb"]
        report.append("| " + " | ".join((label,
            ci(acc["mean"], acc["ci95_background"]),
            ci(u4["mean_skill"], u4["ci95_background"]),
            ci(pop["mean_skill"], pop["ci95_background"]),
            ci(diff["mean_difference"], diff["ci95_background_bootstrap"]),
            ci(fi["mean"], fi["ci95_background"]))) + " |")
    report.extend(["", "## Energy-matched Fisher comparisons", "",
                   "| Regime | Block minus random I_eff | Block minus smooth I_eff | Decision |",
                   "|---|---:|---:|---|"])
    for regime, label in (("IDLIKE", "ID-like"), ("BACKGROUND_OOD", "OOD")):
        r = comparisons[f"{regime}|BLOCK_24_ENERGY_MATCHED_minus_RANDOM_CORNER_ENERGY_MATCHED|I_eff_pre"]
        m = comparisons[f"{regime}|BLOCK_24_ENERGY_MATCHED_minus_MULTIFREQ_SMOOTH_ENERGY_MATCHED|I_eff_pre"]
        report.append("| " + " | ".join((label,
            ci(r["mean_difference"], r["ci95_background_bootstrap"]),
            ci(m["mean_difference"], m["ci95_background_bootstrap"]),
            "PASS" if r["ci95_background_bootstrap"][0] > 0 and m["ci95_background_bootstrap"][0] > 0 else "FAIL")) + " |")
    report.extend(["", "## Baseline functional-class accuracy at N=100 on block schedule", "",
                   "| Regime | V5 recalibrated | V2 | V3 corrected | M4 |", "|---|---:|---:|---:|---:|"])
    for regime, label in (("IDLIKE", "ID-like"), ("BACKGROUND_OOD", "OOD")):
        vals = []
        for estimator in ("V5_RECALIBRATED", "V2_RETRAINED", "V3_CORRECTED", "M4"):
            item = summary["metrics"][f"{regime}|{estimator}|BLOCK_24_ENERGY_MATCHED|N=100|functional_class_accuracy"]
            vals.append(ci(item["mean"], item["ci95_background"]))
        report.append("| " + " | ".join([label, *vals]) + " |")
    report.extend(["", "Ordered-pair accuracy and paired noninferiority to the sinusoidal arm are evaluated separately "
                   "for both orders in `baseline_comparisons.json`. Program-selected effects and oracle-code U1–U4 "
                   "effect diagnostics are distinct rows in `effect_per_background.csv`.", "",
                   "## Gate results", "", "```json", json.dumps(decision, indent=2), "```", "",
                   "No protected test split was opened. The protocol does not change the original fixed-policy or "
                   "post-change active verdicts."])
    (out / "DECISION.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    frozen = ROOT / "results/q1/q1_v5_fixed_confirmatory_dev/training_records.json"
    audit = json.loads((out / "training_and_inference_audit.json").read_text(encoding="utf-8"))
    (out / "training_records.json").write_text(json.dumps({
        "frozen_v5_source_records": json.loads(frozen.read_text(encoding="utf-8")),
        "frozen_v5_source_sha256": sha(frozen),
        "new_train_only_recalibration_fits": audit["training_fits"],
        "new_train_only_calibration": audit["training_only_calibration"],
        "development_data_used_for_selection_or_calibration": False,
        "packaging_script_sha256": sha(Path(__file__)),
    }, indent=2), encoding="utf-8")
    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["postprocessing"] = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha(Path(__file__)),
        "did_not_refit_or_change_protocol": True,
    }
    provenance["deliverable_sha256"] = {
        name: sha(out / name) for name in (
            "config.resolved.json", "split_manifest.json", "training_records.json",
            "train_only_calibration.json", "training_and_inference_audit.json",
            "per_background_per_program.csv", "effect_per_background.csv",
            "policy_action_energy.csv", "action_sequence_hashes.csv",
            "fisher_and_b_error_per_background.csv", "u4_double_intervention_confusions.csv",
            "baseline_comparisons.json", "bootstrap_ci95.json", "summary.json", "DECISION.md")
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/q1_v5_prechange_finalize.py OUTPUT_DIR")
    finalize(Path(sys.argv[1]))
