"""Frozen-checkpoint cross-fitted U4 diagnostics (validation only; no training)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.datasets import rollout, sample_environment, stage_pair
from q1.diagnostics.u4_crossfit import (
    batched_context_jacobian, blocked_inner_indices,
    effect_and_prediction_metrics, ensure_write_once_directory, missing_required_files,
    predict_crossfit_query,
    select_fit_ridge_and_refit, solve_query_refit_upper_bound,
)
from q1.model import TransitionNormalizer
from q1.primitives import PrimitiveBank
from q1.simulator import PROGRAMS, apply_program, response
from q1.train_primitives import _get_m2
from scripts.q1_u4_root_cause import _make_support, _probe_rows


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(file.relative_to(path).as_posix().encode())
        h.update(bytes.fromhex(sha(file)))
    return h.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _context(model, norm: TransitionNormalizer, traj, device: torch.device) -> torch.Tensor:
    support = _make_support(traj.states, traj.actions, norm, device)
    with torch.no_grad():
        return model.encode(support)[0]


def _pred(model, norm, context, probes, device):
    x = torch.as_tensor(probes, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = norm.denormalize_dv(model.executor(norm.normalize_x(x), context.reshape(1, -1).expand(len(x), -1)))
    return y.cpu().numpy().astype(np.float64)


def _flatten_jacobian(jac: np.ndarray, effect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return jac.reshape(-1, jac.shape[-1]), effect.reshape(-1)


def _summarize(rows: list[dict], config: dict) -> dict:
    groups = {}
    z = float(config["confidence_interval_z"])
    for fit_n in config["fit_counts"]:
        for code in config["physical_codes"]:
            for method in ("persistence", "linear_crossfit", "nonlinear_crossfit", "learned_m3",
                           "after_long_context", "query_refit_upper_bound"):
                subset = [r["methods"][method] for r in rows
                          if r["fit_count"] == fit_n and r["physical_code"] == code]
                skills = np.asarray([x["effect_skill"] for x in subset if x["effect_skill"] is not None], dtype=float)
                energies = np.asarray([x["effect_energy_delta_v_squared"] for x in subset], dtype=float)
                errors = np.asarray([x["effect_mse_delta_v_squared"] for x in subset], dtype=float)
                mean_skill = float(skills.mean()) if len(skills) else None
                se = float(skills.std(ddof=1) / math.sqrt(len(skills))) if len(skills) > 1 else 0.0
                groups[f"fit{fit_n}/physical_code{code}/{method}"] = {
                    "background_count": len(subset),
                    "mean_background_effect_skill": mean_skill,
                    "mean_background_effect_skill_95ci": ([mean_skill-z*se, mean_skill+z*se]
                        if mean_skill is not None else None),
                    "ratio_of_mean_effect_mse_to_energy_skill": (
                        float(1.0-errors.mean()/energies.mean()) if len(subset) and energies.mean() > 0 else None),
                    "mean_effect_mse_delta_v_squared": float(errors.mean()) if len(subset) else None,
                    "mean_effect_energy_delta_v_squared": float(energies.mean()) if len(subset) else None,
                    "mean_after_gain": float(np.mean([x["learned_after_gain"] for x in subset
                                                        if x["learned_after_gain"] is not None])) if any(
                        x["learned_after_gain"] is not None for x in subset) else None,
                    "query_refit_is_leaky_upper_bound": method == "query_refit_upper_bound",
                }
    return {"status": "DIAGNOSTIC_ONLY", "aggregates": groups,
            "uncertainty_unit": "validation background (n=16); not transitions or seeds"}


def _classify(rows: list[dict], summary: dict) -> dict:
    # Conservative evidence labels: do not promote diagnostics to a Q1 pass.
    reasons = []
    for code in (0, 2, 3):
        lin = summary["aggregates"].get(f"fit128/physical_code{code}/linear_crossfit", {})
        nonlin = summary["aggregates"].get(f"fit128/physical_code{code}/nonlinear_crossfit", {})
        oracle = summary["aggregates"].get(f"fit128/physical_code{code}/query_refit_upper_bound", {})
        learned = summary["aggregates"].get(f"fit128/physical_code{code}/learned_m3", {})
        lin_ci = lin.get("mean_background_effect_skill_95ci")
        nonlin_ci = nonlin.get("mean_background_effect_skill_95ci")
        learned_ci = learned.get("mean_background_effect_skill_95ci")
        oracle_skill = oracle.get("mean_background_effect_skill")
        transfer_signal = bool(lin_ci and nonlin_ci and lin_ci[0] > 0 and nonlin_ci[0] > 0)
        learned_fails = bool(learned_ci and learned_ci[1] < 0)
        if code == 3 and transfer_signal and learned_fails:
            code_evidence = "U4_OPERATOR_BOTTLENECK_CANDIDATE_DIAGNOSTIC_ONLY"
        elif transfer_signal:
            code_evidence = "CROSSFIT_EFFECT_SIGNAL_DIAGNOSTIC_ONLY"
        else:
            code_evidence = "INCONCLUSIVE"
        reasons.append({"physical_code": code, "linear_crossfit_ci": lin_ci,
                        "nonlinear_crossfit_ci": nonlin_ci,
                        "learned_m3_ci": learned_ci,
                        "query_refit_mean_skill_leaky": oracle_skill,
                        "evidence_label": code_evidence})
    return {"category": "INCONCLUSIVE", "full_q1_pass": False,
            "rule": "positive clustered skill CI indicates signal above zero only; no preregistered accuracy cutoff; mixed code-level evidence stays inconclusive",
            "reason": "U1/U3/U4 do not form one uniform result, and inspected validation cannot establish final acceptance",
            "by_physical_code": reasons}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/q1_u4_crossfit_v2.yaml")
    ap.add_argument("--m2-checkpoint", default="results/q1/cpu_confirmatory_impl/m2/context_model.pt")
    ap.add_argument("--m3-checkpoint", default="results/q1/repair_v1_m3_effect_ablation/beta_0.25/primitives.pt")
    ap.add_argument("--split-config", default="configs/q1_splits.yaml")
    ap.add_argument("--out", default="results/q1/u4_crossfit_v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    config_path, m2_path, m3_path = [ROOT / Path(p) for p in (args.config, args.m2_checkpoint, args.m3_checkpoint)]
    split_path, out = ROOT / args.split_config, ROOT / args.out
    try:
        ensure_write_once_directory(out)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_cfg = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    manifest_path = m2_path.parent.parent / "split_manifest.json"
    legacy_roots = [ROOT / "results/q1/u4_root_cause_v1", ROOT / "results/q1/repair_v1_m3_effect_ablation"]
    legacy_before = {str(p.relative_to(ROOT)): tree_sha(p) for p in legacy_roots if p.exists()}
    missing = missing_required_files([m2_path, m3_path, manifest_path])
    if missing:
        dump_json(out / "blocked.json", {"status": "BLOCKED_MISSING_REQUIRED_LOCAL_ARTIFACTS", "missing": missing})
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Narrow manifest to validation to ensure the script cannot accidentally enumerate test IDs.
    manifest["regions"] = {"validation": manifest["regions"]["validation"]}
    manifest["environment_ids"] = {"validation": manifest["environment_ids"]["validation"]}
    manifest["environment_seeds"] = {"validation": manifest["environment_seeds"]["validation"]}
    if len(manifest["environment_ids"]["validation"]) < int(cfg["validation_backgrounds"]):
        raise ValueError("original split manifest does not contain the configured validation backgrounds")
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    started = time.perf_counter()
    model, norm, m2_data = _get_m2(m2_path, device)
    m3_data = torch.load(m3_path, map_location=device, weights_only=False)
    model_cfg = m2_data["config"]["model"]
    bank = PrimitiveBank(int(model_cfg["context_dim"]), hidden_dim=64).to(device)
    bank.load_state_dict(m3_data["primitives"])
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)
    code_map = m3_data["group_to_code"]
    if set(code_map) != {str(i) for i in range(4)} or sorted(map(int, code_map.values())) != list(range(4)):
        raise ValueError("M3 group_to_code mapping is not bijective")
    rows = []
    for env in range(int(cfg["validation_backgrounds"])):
        before = sample_environment(manifest, "validation", env, split_cfg["parameter_ranges"])
        for code in cfg["physical_codes"]:
            # Pair the same old support, action schedule, and query seed across interventions.
            case_seed = int(cfg["seed"]) + env * 100003
            old, adaptation, query = stage_pair(before, (int(code),), seed=case_seed,
                n_support=int(cfg["old_context_support"]), n_query=max(map(int, cfg["fit_counts"])),
                dt=float(cfg["dt"]), sigma=float(cfg["noise_sigma"]))
            c0 = _context(model, norm, old, device)
            raw_fit_all = _probe_rows(adaptation)
            raw_query = _probe_rows(query)[:int(cfg["query_count"])]
            query_j = batched_context_jacobian(model, norm, c0, raw_query, device)
            c_long = _context(model, norm, rollout(apply_program(before, (int(code),)), n=96,
                seed=case_seed + 900002, dt=float(cfg["dt"]), sigma=float(cfg["noise_sigma"])), device)
            # Truth/query labels are intentionally computed only after fit-only selection and predictions.
            for fit_n in cfg["fit_counts"]:
                raw_fit = raw_fit_all[:int(fit_n)]
                jfit = batched_context_jacobian(model, norm, c0, raw_fit, device)
                true_fit_old = response(before, raw_fit, dt=float(cfg["dt"]))
                true_fit_new = response(apply_program(before, (int(code),)), raw_fit, dt=float(cfg["dt"]))
                effect_fit = true_fit_new - true_fit_old
                jf, yf = _flatten_jacobian(jfit, effect_fit)
                train_i, val_i = blocked_inner_indices(int(fit_n), cfg["inner_validation_fraction"], cfg["inner_gap_fraction"])
                tr_j, tr_y = _flatten_jacobian(jfit[train_i], effect_fit[train_i])
                va_j, va_y = _flatten_jacobian(jfit[val_i], effect_fit[val_i])
                dc, solve_info = select_fit_ridge_and_refit(jf, yf, tr_j, tr_y, va_j, va_y,
                    ridge_factors=cfg["ridge_factors"])
                pred_nl = predict_crossfit_query(model, norm, c0, dc, raw_query, device)
                pred_old = pred_nl["pred_old"]
                linear_effect = np.einsum("nij,j->ni", query_j, dc)
                linear_new = pred_old + linear_effect
                pred_m3 = _pred(model, norm, bank.apply_one(c0, int(code_map[str(int(code))])), raw_query, device)
                pred_long = _pred(model, norm, c_long, raw_query, device)
                # Freeze cross-fit model selection/predictions before opening query labels.
                true_q_old = response(before, raw_query, dt=float(cfg["dt"]))
                true_q_new = response(apply_program(before, (int(code),)), raw_query, dt=float(cfg["dt"]))
                methods = {
                    "persistence": (pred_old, pred_old),
                    "linear_crossfit": (pred_old, linear_new),
                    "nonlinear_crossfit": (pred_nl["pred_old"], pred_nl["pred_new"]),
                    "learned_m3": (pred_old, pred_m3),
                    "after_long_context": (pred_old, pred_long),
                }
                metric_rows = {name: effect_and_prediction_metrics(pred_old=a, pred_new=b,
                    true_old=true_q_old, true_new=true_q_new) for name, (a,b) in methods.items()}
                jq_flat, yq = _flatten_jacobian(query_j, true_q_new - true_q_old)
                dc_oracle, oracle_info = solve_query_refit_upper_bound(jq_flat, yq, ridge=0.0)
                qfit = predict_crossfit_query(model, norm, c0, dc_oracle, raw_query, device)
                metric_rows["query_refit_upper_bound"] = effect_and_prediction_metrics(
                    pred_old=qfit["pred_old"], pred_new=qfit["pred_new"],
                    true_old=true_q_old, true_new=true_q_new)
                oracle_info["excluded_from_feasibility_gate"] = True
                def coverage(probes):
                    speed = np.linalg.norm(probes[:, 2:4], axis=1)
                    action = probes[:, 4:6]
                    eig = np.linalg.eigvalsh(np.cov(action, rowvar=False))
                    return {"speed_p10": float(np.quantile(speed, .10)),
                            "speed_median": float(np.median(speed)),
                            "speed_p90": float(np.quantile(speed, .90)),
                            "action_covariance_eigenvalues": eig.tolist(),
                            "action_covariance_condition": float(eig[-1]/eig[0]) if eig[0] > 0 else None}
                rows.append({"background": env, "environment_id": manifest["environment_ids"]["validation"][env],
                    "physical_code": int(code), "physical_program": [int(code)], "fit_count": int(fit_n),
                    "query_count": len(raw_query), "seed": case_seed, "sigma": cfg["noise_sigma"],
                    "fit_seed": case_seed, "query_seed": case_seed + 2,
                    "fit_action_seed": case_seed + 104730,
                    "fit_noise_seed": case_seed + 209760,
                    "fit_probe_sha256": hashlib.sha256(raw_fit.tobytes()).hexdigest(),
                    "query_probe_sha256": hashlib.sha256(raw_query.tobytes()).hexdigest(),
                    "fit_coverage": coverage(raw_fit), "query_coverage": coverage(raw_query),
                    "fit_query_probe_overlap": False,
                    "query_labels_opened_after_crossfit_freeze": True,
                    "ridge_selection": solve_info, "displacement_norm": float(np.linalg.norm(dc)),
                    "query_refit_diagnostics": oracle_info, "methods": metric_rows})
    summary = _summarize(rows, cfg)
    diagnosis = _classify(rows, summary)
    config_resolved = {**cfg, "m3_group_to_code": code_map, "manifest_used_splits": ["validation"],
                       "test_split_evaluated": False, "training_or_retraining": False}
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(config_resolved, sort_keys=False), encoding="utf-8")
    with (out / "crossfit_per_background.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    dump_json(out / "crossfit_metrics.json", summary)
    dump_json(out / "diagnosis.json", diagnosis)
    p0_rows = [{"background": r["background"], "physical_code": r["physical_code"],
                "fit_count": r["fit_count"], **{k: r["methods"]["nonlinear_crossfit"][k] for k in (
                    "base_error_delta_v_squared", "persistence_error_delta_v_squared",
                    "after_error_delta_v_squared", "effect_energy_delta_v_squared",
                    "effect_mse_delta_v_squared", "effect_skill", "perfect_effect_after_mse_delta_v_squared",
                    "perfect_effect_gain", "learned_after_gain", "perfect_effect_identity_abs_error",
                    "decomposition_max_abs_error_delta_v", "decomposition_identity_pass")}}
               for r in rows]
    dump_json(out / "p0_metric_audit.json", {
        "scope": "crossfit query metrics; effect change distinct from absolute after error",
        "legacy_after_mse_gain_threshold": cfg["legacy_after_mse_gain_threshold"],
        "legacy_gate_for_comparison": "historical 5% after-MSE improvement; diagnostic reference only",
        "gate_status": "DEVELOPMENT_ONLY; not a Q1 pass",
        "per_background": p0_rows,
        "perfect_effect_identity_max_abs_error": max(r["methods"]["nonlinear_crossfit"]["perfect_effect_identity_abs_error"] for r in rows),
        "decomposition_max_abs_error": max(r["methods"]["nonlinear_crossfit"]["decomposition_max_abs_error_delta_v"] for r in rows),
        "per_background_metrics_saved": True})
    csv_rows = []
    for r in rows:
        for method, m in r["methods"].items():
            csv_rows.append({"background": r["background"], "environment_id": r["environment_id"],
                "physical_code": r["physical_code"], "fit_count": r["fit_count"],
                "fit_seed": r["fit_seed"], "fit_action_seed": r["fit_action_seed"],
                "query_seed": r["query_seed"], "method": method,
                "effect_mse": m["effect_mse_delta_v_squared"], "effect_energy": m["effect_energy_delta_v_squared"],
                "effect_skill": m["effect_skill"], "base_error": m["base_error_delta_v_squared"],
                "persistence_error": m["persistence_error_delta_v_squared"], "after_error": m["after_error_delta_v_squared"],
                "perfect_effect_gain": m["perfect_effect_gain"], "after_gain": m["learned_after_gain"],
                "ridge_factor": r["ridge_selection"]["selected_ridge_factor"],
                "ridge_lambda": r["ridge_selection"]["ridge_lambda"], "alpha": 1.0,
                "displacement_norm": r["displacement_norm"],
                "jacobian_condition_number": r["ridge_selection"]["condition_number"],
                "fit_speed_median": r["fit_coverage"]["speed_median"],
                "query_speed_median": r["query_coverage"]["speed_median"],
                "fit_action_condition": r["fit_coverage"]["action_covariance_condition"],
                "query_action_condition": r["query_coverage"]["action_covariance_condition"],
                "query_refit_leaky": method == "query_refit_upper_bound"})
    with (out / "crossfit_per_background.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    legacy_after = {str(p.relative_to(ROOT)): tree_sha(p) for p in legacy_roots if p.exists()}
    if legacy_before != legacy_after:
        raise RuntimeError("legacy protected artifact trees changed during diagnostic run")
    dump_json(out / "provenance.json", {"status": "COMPLETE", "run_kind": cfg["run_kind"],
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "m2_checkpoint": str(m2_path.relative_to(ROOT)), "m2_sha256": sha(m2_path),
        "m3_checkpoint": str(m3_path.relative_to(ROOT)), "m3_sha256": sha(m3_path),
        "split_manifest": str(manifest_path.relative_to(ROOT)), "split_manifest_sha256": sha(manifest_path),
        "config_sha256": sha(config_path), "script_sha256": sha(Path(__file__)),
        "legacy_protected_tree_hashes_before": legacy_before, "legacy_protected_tree_hashes_after": legacy_after,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "torch_threads": torch.get_num_threads(), "wall_seconds": time.perf_counter()-started,
        "test_or_ood_evaluated": False, "model_trained": False, "query_refit_is_label_leakage": True})
    u4_virtual = [r["perfect_effect_gain"] for r in p0_rows
                  if r["physical_code"] == 3 and r["fit_count"] == 128 and r["perfect_effect_gain"] is not None]
    u4_lin = summary["aggregates"]["fit128/physical_code3/linear_crossfit"]
    u4_nonlin = summary["aggregates"]["fit128/physical_code3/nonlinear_crossfit"]
    u4_m3 = summary["aggregates"]["fit128/physical_code3/learned_m3"]
    (out / "gate_report.md").write_text(
        "# U4 cross-fit diagnostic v2\n\n"
        f"**Classification: {diagnosis['category']}** — DEVELOPMENT_ONLY; no FULL_Q1_PASS.\n\n"
        "| View | Evidence on fit=128/query=256 | Interpretation |\n"
        "|---|---|---|\n"
        f"| v1 historical gate | 5% relative after-MSE gain; perfect-effect U4 mean-of-ratios {np.mean(u4_virtual):.2%}, range {min(u4_virtual):.2%}–{max(u4_virtual):.2%} | Historical gate is retained for comparison and fails even for virtual perfect effect on these probes. |\n"
        f"| v2 transformation fidelity | U4 linear skill {u4_lin['mean_background_effect_skill']:.3f} (95% CI {u4_lin['mean_background_effect_skill_95ci'][0]:.3f}–{u4_lin['mean_background_effect_skill_95ci'][1]:.3f}); nonlinear {u4_nonlin['mean_background_effect_skill']:.3f} ({u4_nonlin['mean_background_effect_skill_95ci'][0]:.3f}–{u4_nonlin['mean_background_effect_skill_95ci'][1]:.3f}); M3 {u4_m3['mean_background_effect_skill']:.3f} ({u4_m3['mean_background_effect_skill_95ci'][0]:.3f}–{u4_m3['mean_background_effect_skill_95ci'][1]:.3f}) | Candidate U4 operator-learning bottleneck, diagnostic-only; mixed U1/U3/U4 evidence leaves overall result INCONCLUSIVE. |\n\n"
        "All intervals are clustered by validation background. The query-refit projection uses query labels and is marked QUERY-LABEL LEAKAGE / NONTRANSFER ORACLE; it is excluded from transfer conclusions. This is not an untouched confirmatory evaluation.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
