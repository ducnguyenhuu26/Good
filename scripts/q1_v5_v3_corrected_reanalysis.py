"""Re-score the frozen Phase B V3 baseline with its true marginal-score argmin.

The original Phase B output is read-only and retained as historical evidence.
This produces a side-by-side corrected directory on exactly the same seeds and
deterministic trajectories; it does not inspect protected test data.
"""
from __future__ import annotations

import csv
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from q1.datasets import stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS, DynamicsParams
from q1_v3.program_search import predict_effect, score_program_budgets
from q1_v3.task_inference import fit_task_map
from scripts.q1_v5_fixed_confirmatory import (
    BUDGETS, DT, SIGMA, boot, grouped, load_baselines, paired,
    select_v3_candidate, xy,
)


SRC = ROOT / "results/q1/q1_v5_fixed_confirmatory_dev"
OUT = ROOT / "results/q1/q1_v5_fixed_confirmatory_dev_v3_corrected"
_REUSE_ROWS: Path | None = None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refuse to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SRC / "split_manifest.json").read_text(encoding="utf-8"))
    frozen_summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8"))
    frozen_rows = list(csv.DictReader((SRC / "per_background_per_program_metrics.csv").open(encoding="utf-8")))
    corrected_rows = []
    baseline = load_baselines()["v3"]
    v3action, v3bank, v3ck, v3map = baseline
    programs = [tuple(v3map[c] for c in p) for p in PROGRAMS]
    p2o = {int(k): int(v) for k, v in v3map.items()}
    o2p = {v: k for k, v in p2o.items()}

    regions = () if _REUSE_ROWS else (("IDLIKE", "development_idlike"),
                                      ("BACKGROUND_OOD", "development_background_ood"))
    for region, split_name in regions:
        for bi, row in enumerate(manifest["sets"][split_name]["backgrounds"]):
            p = DynamicsParams(np.asarray(row["B"], float), float(row["d"]),
                               float(row["b"]), float(row["kappa"]))
            seed = int(manifest["sets"][split_name]["seeds"][bi])
            classes, _ = functional_classes(p, dt=DT)
            for pi, program in enumerate(PROGRAMS):
                old, adapt, query = stage_pair(
                    p, program, seed=seed + 500000 + pi * 7919,
                    n_support=96, n_query=256, dt=DT, sigma=SIGMA)
                x0, y0 = xy(old); xa, ya = xy(adapt); xq, yq = xy(query)
                fit = fit_task_map(
                    torch.as_tensor(x0, dtype=torch.float64),
                    torch.as_tensor(y0, dtype=torch.float64), v3action,
                    v3ck["prior_mean"], v3ck["prior_covariance"],
                    dt=DT, sigma=SIGMA, seed=seed + pi, starts=1, maxiter=100)
                all_scored = score_program_budgets(
                    torch.as_tensor(xa, dtype=torch.float64),
                    torch.as_tensor(ya, dtype=torch.float64), fit.theta.detach(),
                    fit.covariance.detach(), v3bank, v3action, programs,
                    BUDGETS, dt=DT, sigma=SIGMA)
                for n in BUDGETS:
                    chosen = select_v3_candidate(all_scored[n])
                    pred = tuple(o2p[c] for c in chosen["program"])
                    qpred = v3_mean_query(xq, chosen["theta_mean"], v3action)
                    corrected_rows.append({
                        "regime": region, "background": f"{region}-{bi:02d}",
                        "true_program": json.dumps(program), "N": n, "model": "V3",
                        "predicted_program": json.dumps(pred),
                        "raw_correct": int(pred == program),
                        "class_correct": int(classes[PROGRAMS.index(pred)] == classes[PROGRAMS.index(program)]),
                        "independent_query_mse": float(np.mean((qpred-yq)**2)),
                        "selected_marginal_score": float(chosen["marginal_score"]),
                        "selected_point_score": float(chosen["point_score"]),
                    })
                if len(program) == 1:
                    code = program[0]
                    delta = predict_effect(torch.as_tensor(xq, dtype=torch.float64),
                                           fit.theta.detach(), v3bank, v3map[code],
                                           v3action, dt=DT).detach().numpy()
                    truth = np.zeros_like(delta)
                    # Effect rows in Phase B remain the same code-conditioned metric;
                    # preserve them verbatim below rather than recomputing the oracle.
                    del truth, delta
            if (bi + 1) % 4 == 0:
                print(f"{region} V3 corrected {bi+1}/{len(manifest['sets'][split_name]['seeds'])}", flush=True)

    if _REUSE_ROWS:
        with _REUSE_ROWS.open(encoding="utf-8") as f:
            corrected_rows = list(csv.DictReader(f))
        for row in corrected_rows:
            row["N"] = int(row["N"])
            for key in ("raw_correct", "class_correct"):
                row[key] = int(row[key])
            for key in ("independent_query_mse", "selected_marginal_score", "selected_point_score"):
                row[key] = float(row[key])
    for row in frozen_rows:
        row["N"] = int(row["N"])
        for key in ("raw_correct", "class_correct"):
            row[key] = int(row[key])
        row["independent_query_mse"] = float(row["independent_query_mse"])
    combined = [r for r in frozen_rows if r["model"] != "V3"] + corrected_rows
    write_csv(OUT / "v3_corrected_rows.csv", corrected_rows)
    write_csv(OUT / "per_background_per_program_metrics.csv", combined)
    # Keep all other frozen outputs byte-identical; the correction only replaces
    # V3 program-ranking predictions and recomputed V3-dependent gates.
    for name in ("config.resolved.json", "split_manifest.json", "training_records.json",
                 "effect_per_background.csv", "direct_context_per_background.csv"):
        shutil.copy2(SRC / name, OUT / name)
    prog_rows = list(csv.DictReader((SRC / "per_program_summary.csv").open(encoding="utf-8")))
    prog_rows = [r for r in prog_rows if r["model"] != "V3"]
    for regime in ("IDLIKE", "BACKGROUND_OOD"):
        for n in BUDGETS:
            for program in PROGRAMS:
                cell = [r for r in corrected_rows if r["regime"] == regime and r["N"] == n
                        and tuple(json.loads(r["true_program"])) == program]
                prog_rows.append({
                    "regime": regime, "model": "V3", "N": n,
                    "true_program": json.dumps(program),
                    "functional_class_accuracy": str(float(np.mean([r["class_correct"] for r in cell]))),
                    "raw_program_accuracy": str(float(np.mean([r["raw_correct"] for r in cell]))),
                    "independent_query_mse": str(float(np.mean([r["independent_query_mse"] for r in cell]))),
                    "class_ci95_background": json.dumps(boot([float(r["class_correct"]) for r in cell], 811+n)),
                    "raw_ci95_background": json.dumps(boot([float(r["raw_correct"]) for r in cell], 812+n)),
                    "query_ci95_background": json.dumps(boot([float(r["independent_query_mse"]) for r in cell], 813+n)),
                })
    write_csv(OUT / "per_program_summary.csv", prog_rows)

    summary = frozen_summary
    metrics = summary["metrics"]
    comparisons = summary["paired_comparisons"]
    for regime in ("IDLIKE", "BACKGROUND_OOD"):
        for n in BUDGETS:
            rows_n = [r for r in corrected_rows if r["regime"] == regime and r["N"] == n]
            bacc = _by_background(rows_n, "class_correct")
            bmse = _by_background(rows_n, "independent_query_mse")
            metrics[f"{regime}|V3|N={n}"] = {
                "functional_class_accuracy": float(np.mean(list(bacc.values()))),
                "class_ci95_background": boot(list(bacc.values()), 920+n),
                "n_backgrounds": len(bacc),
                "independent_query_mse": float(np.mean(list(bmse.values()))),
                "query_mse_ci95_background": boot(list(bmse.values()), 930+n),
            }
        for pair in ((0, 1), (1, 0)):
            cell = [r for r in corrected_rows if r["regime"] == regime and r["N"] == 100
                    and tuple(json.loads(r["true_program"])) == pair]
            grouped = _by_background(cell, "class_correct")
            metrics[f"{regime}|V3|order={pair}"] = {
                "class_accuracy": float(np.mean(list(grouped.values()))),
                "ci95": boot(list(grouped.values()), 940+len(pair)),
            }
        for model_name in ("V5", "V2_RETRAINED", "M4"):
            a = _by_background([r for r in combined if r["regime"] == regime and r["N"] == 100 and r["model"] == model_name], "class_correct")
            b = _by_background([r for r in corrected_rows if r["regime"] == regime and r["N"] == 100], "class_correct")
            comparisons[f"{regime}|{model_name}_minus_V3_class_N100_corrected"] = paired(a, b, 950)
        for pair in ((0, 1), (1, 0)):
            a = _by_background([r for r in corrected_rows if r["regime"] == regime and r["N"] == 100 and tuple(json.loads(r["true_program"])) == pair], "class_correct")
            b = _by_background([r for r in combined if r["regime"] == regime and r["N"] == 100 and r["model"] == "V5" and tuple(json.loads(r["true_program"])) == pair], "class_correct")
            comparisons[f"{regime}|V5_minus_V3_order={pair}_corrected"] = paired(b, a, 960)

    gates = summary["decision"]["gates"]
    ordered_min = all(metrics.get(f"{regime}|V5|order={pair}", {}).get("class_accuracy", 0) >= .8
                      for regime in ("IDLIKE", "BACKGROUND_OOD") for pair in ((0,1),(1,0)))
    ordered_ni = all(comparisons[f"{regime}|V5_minus_V3_order={pair}_corrected"]["ci95_background_bootstrap"][0] >= -.02
                     for regime in ("IDLIKE", "BACKGROUND_OOD") for pair in ((0,1),(1,0)))
    gates["ordered_pair_accuracy_at_least_0.80"] = ordered_min
    gates["ordered_pair_class_noninferior_to_v3_margin_0.02"] = ordered_ni
    summary["decision"]["v3_scorer_corrected"] = True
    summary["decision"]["historical_phase_b_artifacts_unchanged"] = True
    summary["decision"]["v5_fixed_gate"] = "PASS" if all(gates.values()) else "FAIL"
    summary["decision"]["notes"].append("V3 rows were independently re-scored by minimum marginal_score; the original Phase B directory is unchanged.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "DECISION.md").write_text(
        "# Corrected V3 baseline reanalysis\n\n"
        "V3 candidate selection uses `argmin(marginal_score)` on the unchanged Phase B development trajectories. "
        "The historical Phase B directory remains unchanged. See `summary.json` for corrected V3 metrics and recomputed V3-dependent gates.\n\n"
        + json.dumps(summary["decision"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"corrected_v3": {k: v for k, v in metrics.items() if "|V3|N=100" in k},
                      "gates": gates, "verdict": summary["decision"]["v5_fixed_gate"]}, indent=2))


def _by_background(rows: list[dict], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["background"], []).append(float(row[field]))
    return {k: float(np.mean(v)) for k, v in grouped.items()}


def v3_mean_query(x, theta, action):
    from q1_v3.mechanisms import mean_delta_v
    return mean_delta_v(torch.as_tensor(x, dtype=torch.float64), theta, action, DT).detach().numpy()


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-v3-rows", type=Path, default=None,
                        help="Reuse a complete corrected scoring CSV from a failed output-writing attempt")
    _REUSE_ROWS = parser.parse_args().reuse_v3_rows
    main()
