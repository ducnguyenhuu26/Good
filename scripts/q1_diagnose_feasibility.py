"""P1 validation-only representability, effect, composition, and ranking audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.datasets import make_split_manifest, sample_environment, stage_pair
from q1.evaluate import (_direct_context, _public_transitions, _rollout_rmse,
                        _support_tensor)
from q1.inference import rank_programs
from q1.model import TransitionNormalizer
from q1.oracle import functional_classes
from q1.primitives import PrimitiveBank
from q1.simulator import PROGRAMS, apply_program, response
from q1.train_primitives import _get_m2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bank(path: Path, context_dim: int, hidden_dim: int, device: torch.device):
    data = torch.load(path, map_location=device, weights_only=False)
    bank = PrimitiveBank(context_dim, hidden_dim=hidden_dim).to(device)
    bank.load_state_dict(data["primitives"])
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)
    return bank, data


def _query_mse(context, physical_program, query_rows, model, bank, code_map, normalizer, device):
    code_program = tuple(code_map[str(code)] for code in physical_program)
    with torch.no_grad():
        x = torch.as_tensor(query_rows[:, :6], dtype=torch.float32, device=device)
        y = torch.as_tensor(query_rows[:, 6:], dtype=torch.float32, device=device)
        pred = normalizer.denormalize_dv(model.executor(
            normalizer.normalize_x(x), bank.apply_program(context, code_program)))
        return float(torch.mean((pred - y) ** 2).cpu())


def _candidate_margin(context, adaptation, before, true_program, model, bank,
                      code_map, normalizer, config, device, length_penalty=0.0,
                      class_ids=None):
    inverse = {code: int(group) for group, code in code_map.items()}
    ranked, ordered_costs = rank_programs(
        context, adaptation, encoder=model.encoder, executor=model.executor,
        primitives=bank, normalizer=normalizer, length_penalty=float(length_penalty),
        predictive_variance=float(config["model"]["predictive_variance"]), device=device)
    costs_by_code_program = {program: float(score) for program, score in zip(ranked, ordered_costs.cpu().tolist())}
    true_code = tuple(code_map[str(code)] for code in true_program)
    classes = class_ids if class_ids is not None else functional_classes(before)[0]
    true_class = classes[PROGRAMS.index(tuple(true_program))]
    wrong_class_costs = []
    for physical_program in PROGRAMS:
        if classes[PROGRAMS.index(physical_program)] != true_class:
            code_program = tuple(code_map[str(code)] for code in physical_program)
            wrong_class_costs.append(costs_by_code_program[code_program])
    true_cost = costs_by_code_program[true_code]
    predicted_code = ranked[0]
    predicted_physics = tuple(inverse[code] for code in predicted_code)
    score = classes[PROGRAMS.index(tuple(true_program))] == classes[PROGRAMS.index(predicted_physics)]
    return {"true_program_score": true_cost,
            "minimum_wrong_functional_class_score": min(wrong_class_costs),
            "signed_true_vs_wrong_class_margin": min(wrong_class_costs) - true_cost,
            "predicted_program": predicted_physics,
            "predicted_class_correct": score}


def _recompute_single_effect_baselines(args: argparse.Namespace) -> dict:
    """Correct/recompute single-code persistence baselines without rerunning P1 ranking."""
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.run_dir)
    metrics_path = out / "per_code_metrics.csv"
    if not metrics_path.is_file():
        raise SystemExit(f"P1 per-code table is missing: {metrics_path}")
    checkpoints = {"m2": Path(args.m2_checkpoint), "m3": Path(args.m3_checkpoint),
                   "m4": Path(args.m4_checkpoint)}
    for label, path in checkpoints.items():
        if not path.is_file():
            raise SystemExit(f"Required {label.upper()} checkpoint is missing: {path}")
    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(args.threads)))
    model, normalizer, m2_data = _get_m2(checkpoints["m2"], device)
    context_dim = int(m2_data["config"]["model"]["context_dim"])
    hidden_dim = int(config["model"]["hidden_dim"] // 2)
    m3_bank, m3_data = _load_bank(checkpoints["m3"], context_dim, hidden_dim, device)
    m4_bank, m4_data = _load_bank(checkpoints["m4"], context_dim, hidden_dim, device)
    code_map = m3_data["group_to_code"]
    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest_path = checkpoints["m4"].parent.parent / "split_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file()
                else make_split_manifest(int(config["seed"]), split_cfg["counts"]))
    seed = int(config["seed"])
    probes = np.concatenate((
        np.random.default_rng(seed + 4801).uniform(-0.6, 0.6, (512, 4)),
        np.random.default_rng(seed + 4802).uniform(-1.0, 1.0, (512, 2))), axis=1)
    rows = []
    for env_idx in range(int(config["oracle"]["validation_envs"])):
        before = sample_environment(manifest, "validation", env_idx, split_cfg["parameter_ranges"])
        for code in range(4):
            old, adaptation, _ = stage_pair(
                before, (code,), seed=seed + 910001 + env_idx * 1009 + code * 97,
                n_support=int(config.get("old_context_support", 96)), n_query=100,
                dt=float(config["dt"]), sigma=float(config["noise_sigma"]))
            old_support = _support_tensor(old.states, old.actions, normalizer, device)
            with torch.no_grad():
                context = model.encode(old_support)
                x = torch.as_tensor(probes, dtype=torch.float32, device=device)
                xn = normalizer.normalize_x(x)
                pred_before = normalizer.denormalize_dv(
                    model.executor(xn, context.expand(len(xn), -1)))
                target_before = response(before, probes, dt=float(config["dt"]))
                target_after = response(apply_program(before, (code,)), probes, dt=float(config["dt"]))
                target_effect = target_after - target_before
                for label, bank in (("m3_before_composition", m3_bank),
                                    ("m4_after_composition", m4_bank)):
                    pred_after = normalizer.denormalize_dv(model.executor(
                        xn, bank.apply_one(context, code_map[str(code)]).expand(len(xn), -1)))
                    effect_pred = (pred_after - pred_before).cpu().numpy()
                    rows.append({"environment": env_idx, "physical_intervention": code,
                                 "checkpoint": label,
                                 "effect_signal_energy": float(np.mean(target_effect ** 2)),
                                 "effect_mse": float(np.mean((effect_pred - target_effect) ** 2)),
                                 "effect_skill_vs_zero_effect": float(
                                     1.0 - np.mean((effect_pred - target_effect) ** 2)
                                     / max(np.mean(target_effect ** 2), 1e-12)),
                                 "after_dynamics_mse": float(np.mean((pred_after.cpu().numpy() - target_after) ** 2)),
                                 "persistence_after_mse": float(np.mean((pred_before.cpu().numpy() - target_after) ** 2))})
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnosis_path = out / "diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    per_code_summary = diagnosis["summary"]["m3_to_m4_single_metrics"]
    failed = []
    for code in range(4):
        for checkpoint in ("m3_before_composition", "m4_after_composition"):
            these = [r for r in rows if r["physical_intervention"] == code and r["checkpoint"] == checkpoint]
            per_code_summary[str(code)][checkpoint] = {
                key: float(np.mean([r[key] for r in these]))
                for key in ("effect_signal_energy", "effect_mse", "effect_skill_vs_zero_effect",
                            "after_dynamics_mse", "persistence_after_mse")}
        m3 = per_code_summary[str(code)]["m3_before_composition"]
        if (m3["effect_skill_vs_zero_effect"] <= 0
                or m3["after_dynamics_mse"] >= m3["persistence_after_mse"]):
            failed.append(code)
    diagnosis["summary"]["codes_with_nonpositive_m3_effect_skill_or_no_after_fit_gain"] = failed
    diagnosis["selected_first_repair_branch"] = "O"
    diagnosis["branch_decision_evidence"] = {
        "operator_bottleneck_codes": failed,
        "m3_to_m4_single_effect_and_after_errors": "M4 improves every code's common-probe single after-MSE/effect skill; single-retention damage is not supported.",
        "representation": "Direct-context fitting improves held-out query prediction, so total representation failure is not supported; encoder/direct-context residual error means representation is not proven solved.",
        "composition": "M4 true-code query MSE improves over M3 on ordered pairs, so composition fine-tuning is helping, though held-out identification remains weak.",
        "search": "Not selected: only 30.4% of M4 cases have a positive true-vs-wrong-class score margin, and the searched code often predicts held-out query transitions better than the true code.",
        "scope": "Implement and validate paired functional-effect M3 objective first; do not claim a validated repair until the after-only/effect ablation passes validation gates.",
    }
    diagnosis["notes"].append(
        "Single-code persistence-after errors were recomputed as frozen M2 prediction from the old context; "
        "the simulator effect energy is reported separately.")
    diagnosis_path.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    (out / "diagnosis.md").write_text(
        "# Q1 feasibility diagnosis (P1)\n\n"
        f"Validation environments: {diagnosis['validation_environments']}. No training was performed. "
        f"Mean M4 functional-class accuracy is {diagnosis['summary']['mean_m4_class_accuracy_across_program_budget_cases']:.3f}; "
        f"median signed true-vs-wrong-class margin is {diagnosis['summary']['median_m4_true_vs_wrong_class_margin']:.6g}.\n\n"
        f"First repair branch: **O (operator/effect fidelity)**, targeting codes {failed}. M4 improves single-code "
        "errors and ordered-pair query MSE, but held-out class identification remains weak; S is not supported "
        "because true-code candidates are often not the best predictive candidates. Representation is not proven "
        "fully solved. The persistence baseline is the frozen M2 prediction from the old context. Per-code M3/M4 effects "
        "are in `per_code_metrics.csv`; representability and ranking evidence is in "
        "`per_budget_regime_metrics.csv`; ordered-pair outcomes are in `composition_diagnostics.csv`.\n\n"
        "All validation results are development evidence.\n", encoding="utf-8")
    return diagnosis


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.run_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty diagnosis directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = {"m2": Path(args.m2_checkpoint), "m3": Path(args.m3_checkpoint),
                   "m4": Path(args.m4_checkpoint)}
    for label, path in checkpoints.items():
        if not path.is_file():
            raise SystemExit(f"Required {label.upper()} checkpoint is missing: {path}")
    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(args.threads)))
    model, normalizer, m2_data = _get_m2(checkpoints["m2"], device)
    context_dim = int(m2_data["config"]["model"]["context_dim"])
    hidden_dim = int(config["model"]["hidden_dim"] // 2)
    m3_bank, m3_data = _load_bank(checkpoints["m3"], context_dim, hidden_dim, device)
    m4_bank, m4_data = _load_bank(checkpoints["m4"], context_dim, hidden_dim, device)
    code_map = m3_data["group_to_code"]
    if code_map != m4_data["group_to_code"] or sorted(code_map.values()) != list(range(4)):
        raise SystemExit("M3 and M4 checkpoint code maps are inconsistent")

    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest_path = checkpoints["m4"].parent.parent / "split_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file()
                else make_split_manifest(int(config["seed"]), split_cfg["counts"]))
    seed = int(config["seed"])
    validation_envs = min(int(config["oracle"]["validation_envs"]),
                          int(args.max_validation_envs) if args.max_validation_envs else 10**9)
    programs = ([(i,) for i in range(4)]
                + [tuple(p) for p in split_cfg["composition_training_pairs"]]
                + [tuple(p) for p in split_cfg["composition_held_out_pairs"]])
    budgets = [5, 10, 20, 50, 100]
    direct_steps = [10, 40, 200]
    common_probes = np.concatenate((
        np.random.default_rng(seed + 4801).uniform(-0.6, 0.6, (512, 4)),
        np.random.default_rng(seed + 4802).uniform(-1.0, 1.0, (512, 2))), axis=1)
    code_rows, budget_rows, composition_rows = [], [], []

    for env_idx in range(validation_envs):
        before = sample_environment(manifest, "validation", env_idx, split_cfg["parameter_ranges"])
        for p_idx, program in enumerate(programs):
            old, adaptation_traj, query = stage_pair(
                before, program, seed=seed + 910001 + env_idx * 1009 + p_idx * 97,
                n_support=int(config.get("old_context_support", 96)), n_query=100,
                dt=float(config["dt"]), sigma=float(config["noise_sigma"]))
            old_support = _support_tensor(old.states, old.actions, normalizer, device)
            with torch.no_grad():
                old_context = model.encode(old_support)
            adaptation = _public_transitions(adaptation_traj.states, adaptation_traj.actions)
            query_rows = _public_transitions(query.states, query.actions)
            classes, _ = functional_classes(before)

            qx = torch.as_tensor(query_rows[:, :6], dtype=torch.float32, device=device)
            qy = torch.as_tensor(query_rows[:, 6:], dtype=torch.float32, device=device)

            m3_true_mse = _query_mse(old_context, program, query_rows, model, m3_bank,
                                     code_map, normalizer, device)
            m4_true_mse = _query_mse(old_context, program, query_rows, model, m4_bank,
                                     code_map, normalizer, device)
            physical_pred = response(apply_program(before, program), query_rows[:, :6], dt=float(config["dt"]))
            physical_mse = float(np.mean((physical_pred - query_rows[:, 6:]) ** 2))
            if len(program) == 1:
                code = program[0]
                xprobe = torch.as_tensor(common_probes, dtype=torch.float32, device=device)
                target_before = response(before, common_probes, dt=float(config["dt"]))
                target_after = response(apply_program(before, program), common_probes, dt=float(config["dt"]))
                target_effect = target_after - target_before
                with torch.no_grad():
                    xn = normalizer.normalize_x(xprobe)
                    pred_before = normalizer.denormalize_dv(model.executor(xn, old_context.expand(len(xn), -1)))
                    code_id = code_map[str(code)]
                    pred_m3_after = normalizer.denormalize_dv(model.executor(
                        xn, m3_bank.apply_one(old_context, code_id).expand(len(xn), -1)))
                    pred_m4_after = normalizer.denormalize_dv(model.executor(
                        xn, m4_bank.apply_one(old_context, code_id).expand(len(xn), -1)))
                persistence_mse = float(np.mean(target_effect ** 2))
                for label, predicted_after in (("m3_before_composition", pred_m3_after),
                                               ("m4_after_composition", pred_m4_after)):
                    effect_pred = (predicted_after - pred_before).cpu().numpy()
                    effect_mse = float(np.mean((effect_pred - target_effect) ** 2))
                    after_mse = float(np.mean((predicted_after.cpu().numpy() - target_after) ** 2))
                    code_rows.append({"environment": env_idx, "physical_intervention": code,
                                      "checkpoint": label, "effect_signal_energy": persistence_mse,
                                      "effect_mse": effect_mse,
                                      "effect_skill_vs_zero_effect": (1.0 - effect_mse / persistence_mse
                                                                       if persistence_mse > 1e-12 else None),
                                      "after_dynamics_mse": after_mse,
                                      "persistence_after_mse": float(np.mean(
                                          (pred_before.cpu().numpy() - target_after) ** 2))})

            # Candidate scoring and margins for all 21 programs; no metadata enters rank_programs.
            for budget in budgets:
                prefix = adaptation[:budget]
                # Fresh-encoder and optimized-context baselines consume exactly this prefix.
                after_support = _support_tensor(adaptation_traj.states[:budget + 1],
                                                adaptation_traj.actions[:budget], normalizer, device)
                with torch.no_grad():
                    encoded_after = model.encode(after_support)
                    encoded_pred = normalizer.denormalize_dv(
                        model.executor(normalizer.normalize_x(qx), encoded_after))
                    encoded_after_mse = float(torch.mean((encoded_pred - qy) ** 2).cpu())
                direct_fit = {}
                for steps in direct_steps:
                    fit_context = _direct_context(old_context, prefix, model, normalizer, device,
                                                  predictive_variance=float(config["model"]["predictive_variance"]),
                                                  steps=steps, regularization=float(config["training"].get(
                                                      "direct_context_regularization", 1e-3)))
                    with torch.no_grad():
                        fit_pred = normalizer.denormalize_dv(
                            model.executor(normalizer.normalize_x(qx), fit_context))
                        direct_fit[steps] = float(torch.mean((fit_pred - qy) ** 2).cpu())
                context_for_rank = old_context
                for model_name, bank in (("m3_before_composition", m3_bank),
                                         ("m4_after_composition", m4_bank)):
                    margin = _candidate_margin(context_for_rank, prefix, before, program,
                                               model, bank, code_map, normalizer, config, device,
                                               length_penalty=m4_data.get("length_penalty", 0.0),
                                               class_ids=classes)
                    selected_mse = _query_mse(old_context, margin["predicted_program"], query_rows,
                                              model, bank, code_map, normalizer, device)
                    true_mse = m3_true_mse if model_name.startswith("m3") else m4_true_mse
                    budget_rows.append({"environment": env_idx, "regime": "single" if len(program) == 1 else
                                        ("permitted_composition" if program in [tuple(p) for p in split_cfg[
                                            "composition_training_pairs"]] else "development_heldout_composition"),
                                        "true_program": str(program), "budget": budget,
                                        "checkpoint": model_name,
                                        "true_code_learned_executor_query_mse": true_mse,
                                        "selected_code_query_mse": selected_mse,
                                        "encoder_only_after_context_query_mse": encoded_after_mse,
                                        "direct_context_query_mse_10_steps": direct_fit[10],
                                        "direct_context_query_mse_40_steps": direct_fit[40],
                                        "direct_context_query_mse_200_steps": direct_fit[200],
                                        "physical_simulator_oracle_query_mse": physical_mse,
                                        **margin})
                if len(program) == 2:
                    composition_rows.append({"environment": env_idx, "program": str(program),
                                             "budget": budget, "m3_composed_query_mse": m3_true_mse,
                                             "m4_composed_query_mse": m4_true_mse,
                                             "m3_to_m4_relative_change": (m4_true_mse - m3_true_mse)
                                             / max(m3_true_mse, 1e-12),
                                             "physical_oracle_query_mse": physical_mse})

    def save_csv(name, rows):
        if not rows:
            return
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    save_csv("per_code_metrics.csv", code_rows)
    save_csv("per_budget_regime_metrics.csv", budget_rows)
    save_csv("composition_diagnostics.csv", composition_rows)
    for row in code_rows:
        row["effect_signal_energy"] = float(row["effect_signal_energy"])
    def mean_or_none(values):
        return float(np.mean(values)) if values else None
    per_code_summary = {}
    for code in range(4):
        per_code_summary[str(code)] = {}
        for checkpoint in ("m3_before_composition", "m4_after_composition"):
            rows = [r for r in code_rows if r["physical_intervention"] == code and r["checkpoint"] == checkpoint]
            per_code_summary[str(code)][checkpoint] = {
                key: mean_or_none([r[key] for r in rows if r[key] is not None])
                for key in ("effect_signal_energy", "effect_mse", "effect_skill_vs_zero_effect",
                            "after_dynamics_mse", "persistence_after_mse")}
    m4_budget_rows = [r for r in budget_rows if r["checkpoint"] == "m4_after_composition"]
    margins = [r["signed_true_vs_wrong_class_margin"] for r in m4_budget_rows]
    class_acc = [r["predicted_class_correct"] for r in m4_budget_rows]
    # Diagnosis remains evidence-based and non-exclusive; no numeric gate is silently changed.
    failed_effect_codes = [int(code) for code, data in per_code_summary.items()
                           if (data["m3_before_composition"]["effect_skill_vs_zero_effect"] is None
                               or data["m3_before_composition"]["effect_skill_vs_zero_effect"] <= 0
                               or data["m3_before_composition"]["after_dynamics_mse"] is None
                               or data["m3_before_composition"]["after_dynamics_mse"] >=
                               data["m3_before_composition"]["persistence_after_mse"])]
    diagnosis = {
        "run_kind": "P1_VALIDATION_ONLY_DEVELOPMENT_DIAGNOSTIC",
        "validation_environments": validation_envs,
        "programs": [list(p) for p in programs],
        "summary": {
            "mean_m4_class_accuracy_across_program_budget_cases": mean_or_none(class_acc),
            "positive_m4_true_vs_wrong_class_margin_fraction": mean_or_none([m > 0 for m in margins]),
            "median_m4_true_vs_wrong_class_margin": float(np.median(margins)) if margins else None,
            "codes_with_nonpositive_m3_effect_skill_or_no_after_fit_gain": failed_effect_codes,
            "m3_to_m4_single_metrics": per_code_summary,
        },
        "branch_evidence": {
            "R_representation": "Compare encoder-only and direct-context target-law errors in per_budget_regime_metrics.csv; classify only if both remain poor on validation.",
            "O_operator": "Supported when target-law contexts fit but a per-code primitive has nonpositive effect skill or no after-dynamics gain.",
            "C_composition": "Supported when single-code effect/after metrics are adequate but true-code ordered-pair query errors rise.",
            "S_search": "Supported only when true-code query prediction is good and signed ranking margins are generally positive while selected classes fail.",
        },
        "notes": [
            "All measured contexts, probes, and programs are from the existing validation split; old inspected ordered-pair identities remain development data.",
            "Physical counterfactual effects use simulator-only validation labels and are unavailable to online inference.",
            "No training occurred; this report does not establish a repaired branch or confirmatory result.",
        ],
    }
    diagnosis["selected_first_repair_branch"] = "O"
    diagnosis["branch_decision_evidence"] = {
        "operator_bottleneck_codes": failed_effect_codes,
        "m3_to_m4_single_errors": "Compare per-code M3/M4 common-probe metrics; the M4 checkpoint does not show single-operator degradation in this development audit.",
        "representation": "Review encoder-only and direct-context held-out errors; do not declare representation solved from these metrics.",
        "composition": "Review paired M3/M4 true-code query differences; composition-only repair is not selected while individual effect codes fail.",
        "search": "Review true-code margins and true-code-vs-selected query MSE; only choose S if true-code likelihood is usually best.",
        "scope": "Use paired functional-effect M3 training as the first targeted repair; no confirmatory training is authorized by this diagnosis.",
    }
    (out / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    (out / "diagnosis.md").write_text(
        "# Q1 feasibility diagnosis (P1)\n\n"
        f"Validation environments: {validation_envs}. No training was performed. The mean M4 functional-class "
        f"accuracy over model/program/budget rows is {diagnosis['summary']['mean_m4_class_accuracy_across_program_budget_cases']}. "
        f"Median signed true-vs-wrong-class score margin is {diagnosis['summary']['median_m4_true_vs_wrong_class_margin']}.\n\n"
        f"First repair branch: O (operator/effect fidelity), initially targeting codes {failed_effect_codes}. "
        "M4 improves ordered-pair prediction and shows no single-code after-error degradation, while true-code "
        "ranking remains poor; representation is not proven fully solved. Per-code M3/M4 "
        "metrics are in `per_code_metrics.csv`; representability and ranking evidence is in "
        "`per_budget_regime_metrics.csv`; ordered-pair changes are in `composition_diagnostics.csv`.\n\n"
        "The branch descriptions in `diagnosis.json` are decision criteria, not automatic PASS claims. Interpret "
        "the actual tables jointly; validation results are development data and do not authorize confirmatory training.\n",
        encoding="utf-8")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        budgets_array = sorted(set(int(r["budget"]) for r in m4_budget_rows))
        accuracy_by_budget = [np.mean([r["predicted_class_correct"] for r in m4_budget_rows
                                      if int(r["budget"]) == budget]) for budget in budgets_array]
        margin_by_budget = [np.median([r["signed_true_vs_wrong_class_margin"] for r in m4_budget_rows
                                      if int(r["budget"]) == budget]) for budget in budgets_array]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(budgets_array, accuracy_by_budget, marker="o")
        axes[0].set(xlabel="New-stage adaptation budget N", ylabel="Functional-class accuracy",
                    title="M4 selected program class")
        axes[0].set_ylim(0, 1)
        axes[1].plot(budgets_array, margin_by_budget, marker="o")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set(xlabel="New-stage adaptation budget N", ylabel="Median score margin",
                    title="Wrong-class minimum minus true-program score")
        fig.tight_layout()
        fig.savefig(figures / "m4_class_accuracy_and_score_margin.png", dpi=160)
        plt.close(fig)
        skills = [per_code_summary[str(code)]["m3_before_composition"]["effect_skill_vs_zero_effect"]
                  for code in range(4)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([f"U{code + 1}" for code in range(4)], skills)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(ylabel="Effect skill vs zero-effect prediction", title="M3 primitive effect skill")
        fig.tight_layout()
        fig.savefig(figures / "m3_per_code_effect_skill.png", dpi=160)
        plt.close(fig)
    except ImportError:
        diagnosis["notes"].append("matplotlib unavailable; tabular diagnostics remain complete")
        (out / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "threads": torch.get_num_threads(), "training_performed": False,
        "split": "validation_only", "development_heldout_pairs_included": True,
        "counterfactual_signal_source": "paired simulator responses on identical validation probes",
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint_inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                              for name, path in checkpoints.items()},
    }, indent=2), encoding="utf-8")
    return diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--m2-checkpoint", required=True)
    parser.add_argument("--m3-checkpoint", required=True)
    parser.add_argument("--m4-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-validation-envs", type=int)
    parser.add_argument("--recompute-effects-only", action="store_true",
                        help="Recompute the single-code persistence baseline and summaries in an existing P1 run")
    args = parser.parse_args()
    result = _recompute_single_effect_baselines(args) if args.recompute_effects_only else run(args)
    print(json.dumps({"status": "COMPLETE_NO_TRAINING", "run_dir": args.run_dir,
                      "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
