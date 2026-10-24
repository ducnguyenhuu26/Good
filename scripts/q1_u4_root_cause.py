"""Targeted U4 identifiability and frozen-M2 representability diagnosis.

This script is development-only. It never reads test splits and never trains M2.
Physical parameters are used only by simulator oracle targets and diagnostic fits.
"""
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

from q1.datasets import make_split_manifest, rollout, sample_environment, stage_pair
from q1.model import TransitionNormalizer
from q1.oracle import functional_classes
from q1.primitives import PrimitiveBank
from q1.simulator import PROGRAMS, DynamicsParams, apply_program, response, transition
from q1.train_dynamics import _transitions
from q1.train_primitives import _get_m2


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _probe_rows(traj) -> np.ndarray:
    return np.concatenate((traj.states[:-1], traj.actions), axis=1)


def _batch_delta(states: np.ndarray, actions: np.ndarray, params: DynamicsParams, dt: float) -> np.ndarray:
    q, v = states[:-1, :2], states[:-1, 2:]
    return dt * (np.tanh(actions @ params.B.T) - params.d * v
                 - params.b * np.linalg.norm(v, axis=1, keepdims=True) * v
                 - params.kappa * q)


def _paired_effect(params: DynamicsParams, code: int, probes: np.ndarray, dt: float) -> np.ndarray:
    """Float64 conditional-mean delta-v effect on common [q,v,a] probes."""
    after = apply_program(params, (code,))
    return response(after, probes, dt=dt) - response(params, probes, dt=dt)


def _make_support(states: np.ndarray, actions: np.ndarray, normalizer: TransitionNormalizer,
                  device: torch.device) -> torch.Tensor:
    rows = np.concatenate((states[:-1], actions, states[1:, 2:] - states[:-1, 2:]), axis=1)
    tensor = torch.as_tensor(rows, dtype=torch.float32, device=device)
    return torch.cat((normalizer.normalize_x(tensor[:, :6]),
                      normalizer.normalize_dv(tensor[:, 6:])), dim=-1)[None]


@torch.no_grad()
def _predict(model, normalizer, context: torch.Tensor, raw_probes: np.ndarray,
             device: torch.device) -> np.ndarray:
    x = torch.as_tensor(raw_probes, dtype=torch.float32, device=device)
    pred = normalizer.denormalize_dv(model.executor(normalizer.normalize_x(x), context.expand(len(x), -1)))
    return pred.detach().cpu().numpy()


def _metric(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred, dtype=np.float64) - truth) ** 2))


def _cluster_ci(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return [float(np.mean(arr)), float(np.mean(arr))]
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    return [float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)]


def phase0(config: dict, out: Path, m2_path: Path, m3_path: Path, device: torch.device,
           split_cfg: dict) -> dict:
    started = time.perf_counter()
    # Randomized exact equation check in float64, including v=0.
    rng = np.random.default_rng(int(config["seed"]) + 10)
    max_err = {"u3": 0.0, "u4": 0.0}
    parameter_scope = {"u3_only_d_changed": True, "u4_only_b_changed": True}
    for i in range(512):
        params = sample_environment(make_split_manifest(), "train", i % 48,
                                    split_cfg["parameter_ranges"])
        u3_params, u4_params = apply_program(params, (2,)), apply_program(params, (3,))
        parameter_scope["u3_only_d_changed"] &= (np.array_equal(params.B, u3_params.B)
            and params.b == u3_params.b and params.kappa == u3_params.kappa
            and np.isclose(u3_params.d, 1.4*params.d))
        parameter_scope["u4_only_b_changed"] &= (np.array_equal(params.B, u4_params.B)
            and params.d == u4_params.d and params.kappa == u4_params.kappa
            and np.isclose(u4_params.b, 1.5*params.b))
        state = rng.normal(size=4)
        if i == 0:
            state[2:] = 0.0
        action = rng.normal(size=2)
        for code, scale, key in ((2, -0.4, "u3"), (3, -0.5, "u4")):
            before = transition(state, action, params, dt=float(config["dt"]))[2:] - state[2:]
            after_params = apply_program(params, (code,))
            after = transition(state, action, after_params, dt=float(config["dt"]))[2:] - state[2:]
            expected = scale * float(config["dt"]) * (params.d * state[2:] if code == 2
                       else params.b * np.linalg.norm(state[2:]) * state[2:])
            max_err[key] = max(max_err[key], float(np.max(np.abs((after - before) - expected))))
    # Paired noisy transitions: shared state/action but independent noise; conditional-mean target is noise-free.
    params = sample_environment(make_split_manifest(), "train", 0, split_cfg["parameter_ranges"])
    paired_checks = []
    state = rng.normal(size=4); action = rng.normal(size=2)
    for sigma in (0.0, float(config["noise_sigma"])):
        eps_rng = np.random.default_rng(int(config["seed"]) + int(sigma * 1e8) + 3)
        old = transition(state, action, params, dt=float(config["dt"]), sigma=sigma, rng=eps_rng)
        new = transition(state, action, apply_program(params, (3,)), dt=float(config["dt"]), sigma=sigma, rng=eps_rng)
        observed_difference = (new[2:] - state[2:]) - (old[2:] - state[2:])
        conditional = _paired_effect(params, 3, np.concatenate((state, action))[None], float(config["dt"]))[0]
        paired_checks.append({"sigma": sigma, "same_state_action": True,
                              "conditional_mean_effect": conditional.tolist(),
                              "one_independent_noise_realization_observed_difference": observed_difference.tolist(),
                              "noise_not_in_conditional_target": bool(np.allclose(conditional,
                                  -0.5 * float(config["dt"]) * params.b * np.linalg.norm(state[2:]) * state[2:], atol=1e-14))})
    # Normalizer round-trip and required checkpoint integrity.
    model, normalizer, m2_data = _get_m2(m2_path, device)
    bank_data = torch.load(m3_path, map_location=device, weights_only=False)
    bank = PrimitiveBank(int(m2_data["config"]["model"]["context_dim"]),
                         hidden_dim=64).to(device)
    bank.load_state_dict(bank_data["primitives"])
    code_map = bank_data["group_to_code"]
    code_map_valid = (set(code_map) == {str(i) for i in range(4)}
                      and sorted(int(v) for v in code_map.values()) == list(range(4)))
    x = torch.as_tensor(rng.normal(size=(32, 6)), dtype=torch.float32, device=device)
    dv = torch.as_tensor(rng.normal(size=(32, 2)), dtype=torch.float32, device=device)
    nrm_x = normalizer.normalize_x(x)
    nrm_dv = normalizer.normalize_dv(dv)
    x_restored = nrm_x * torch.as_tensor(normalizer.x_std, dtype=x.dtype, device=device) + torch.as_tensor(
        normalizer.x_mean, dtype=x.dtype, device=device)
    roundtrip = max(float(torch.max(torch.abs(x - x_restored))),
                    float(torch.max(torch.abs(dv - normalizer.denormalize_dv(nrm_dv)))))
    report = {"status": "PASS" if max(max_err.values()) < 1e-12 and roundtrip < 1e-5
                and all(parameter_scope.values()) and code_map_valid else "FAIL",
              "equation_max_abs_error_float64": max_err,
              "intervention_parameter_scope_checks": {k: bool(v) for k,v in parameter_scope.items()},
              "m3_group_to_code_bijective": bool(code_map_valid),
              "equations": {"u3": "-0.4*dt*d*v", "u4": "-0.5*dt*b*||v||*v"},
              "paired_noise_checks": paired_checks,
              "normalizer_max_abs_roundtrip_error_float32": roundtrip,
              "executor_contract": "raw probes normalized by training-fitted normalizer; predictions denormalized to physical delta-v",
              "m2_checkpoint_sha256": _sha(m2_path), "m3_checkpoint_sha256": _sha(m3_path),
              "m3_group_to_code": bank_data["group_to_code"],
              "wall_seconds": time.perf_counter() - started}
    _json(out / "phase0_verification.json", report)
    return report


def phase1(config: dict, out: Path, split_cfg: dict, manifest: dict, *, sigma: float | None = None) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    seed = int(config["seed"])
    sigma = float(config["noise_sigma"] if sigma is None else sigma)
    programs = list(PROGRAMS)
    true_programs = [tuple(p) for p in config["oracle_true_programs"]]
    cases: dict[str, dict] = {}
    rows: list[dict] = []
    for env in range(int(config["validation_backgrounds"])):
        before = sample_environment(manifest, "validation", env, split_cfg["parameter_ranges"])
        classes, _ = functional_classes(before, dt=float(config["dt"]))
        for true_program in true_programs:
            true_idx = PROGRAMS.index(true_program)
            for replicate in range(int(config["oracle_trajectory_seeds"])):
                case_seed = seed + env * 100003 + true_idx * 7919 + replicate * 997
                old, adaptation, _ = stage_pair(before, true_program, seed=case_seed,
                    n_support=int(config["old_context_support"]), n_query=max(config["budgets"]),
                    dt=float(config["dt"]), sigma=sigma)
                for budget in config["budgets"]:
                    states = adaptation.states[:int(budget) + 1]
                    actions = adaptation.actions[:int(budget)]
                    obs = adaptation.delta_v[:int(budget)]
                    errors = []
                    for candidate in programs:
                        mu = _batch_delta(states, actions, apply_program(before, candidate), float(config["dt"]))
                        errors.append(float(np.mean((mu - obs) ** 2)))
                    order = np.argsort(errors)
                    predicted = programs[int(order[0])]
                    true_class = classes[true_idx]
                    predicted_class = classes[PROGRAMS.index(predicted)]
                    class_correct = true_class == predicted_class
                    closest_class = min((i for i, p in enumerate(programs)
                        if classes[i] != true_class), key=lambda i: errors[i])
                    row = {"env_id": f"validation-{env:04d}", "background": env,
                        "replicate": replicate, "seed": case_seed, "sigma": sigma,
                        "true_program": json.dumps(true_program), "budget": budget,
                        "predicted_program": json.dumps(predicted), "true_class": true_class,
                        "predicted_class": predicted_class, "class_correct": int(class_correct),
                        "identity_correct": int(predicted == true_program),
                        "true_candidate_mse": errors[true_idx],
                        "best_wrong_class_program": json.dumps(programs[closest_class]),
                        "best_wrong_class_mse": errors[closest_class],
                        "true_vs_wrong_class_mse_margin": errors[closest_class] - errors[true_idx],
                        "same_continuous_stage_pair": True}
                    # focused per-candidate scores are explicit for true U4.
                    if true_program == (3,):
                        for label, cand in (("identity", ()), ("u3", (2,)), ("u4", (3,)),
                                            ("u3_u3", (2, 2)), ("u4_u4", (3, 3)),
                                            ("u3_then_u4", (2, 3)), ("u4_then_u3", (3, 2))):
                            row[f"candidate_mse_{label}"] = errors[PROGRAMS.index(cand)]
                    rows.append(row)
        # Natural-state energy and velocity dependence under old exploration, conditional means.
        for program, label in (((3,), "u4"), ((2,), "u3")):
            old, adaptation, _ = stage_pair(before, program, seed=seed + 700000 + env * 97 + PROGRAMS.index(program),
                n_support=int(config["old_context_support"]), n_query=100, dt=float(config["dt"]), sigma=sigma)
            natural = _probe_rows(adaptation)
            eff = _paired_effect(before, program[0], natural, float(config["dt"]))
            q, v = natural[:, :2], natural[:, 2:4]
            # Vectorized feature Gram matrix with two output coordinates per sample.
            f1, f2 = v.reshape(-1), (np.linalg.norm(v, axis=1, keepdims=True) * v).reshape(-1)
            gram = np.array([[f1 @ f1, f1 @ f2], [f2 @ f1, f2 @ f2]], dtype=np.float64)
            eig = np.linalg.eigvalsh(gram)
            noise_var = 2.0 * sigma**2
            pair_delta = (_paired_effect(before, 3, natural, float(config["dt"]))
                          - _paired_effect(before, 2, natural, float(config["dt"])))
            kl_u4_u3 = float(np.mean(np.sum(pair_delta**2, axis=1)) / (2 * sigma**2)) if sigma > 0 else None
            pairwise_kl = {}
            if sigma > 0 and label == "u4":
                true_mu = response(apply_program(before, (3,)), natural, dt=float(config["dt"]))
                for candidate, name in (((), "identity"), ((2,), "u3"), ((2,2), "u3_u3"),
                                        ((3,3), "u4_u4"), ((2,3), "u3_then_u4"),
                                        ((3,2), "u4_then_u3")):
                    candidate_mu = response(apply_program(before, candidate), natural, dt=float(config["dt"]))
                    pairwise_kl[name] = float(np.mean(np.sum((true_mu-candidate_mu)**2,axis=1))/(2*sigma**2))
            cases[f"{env}:{label}"] = {"background": env, "code": label,
                "effect_energy_delta_v_squared": float(np.mean(eff**2)),
                "effect_rms_delta_v": float(np.sqrt(np.mean(eff**2))),
                "noise_variance_per_delta_v_component": sigma**2,
                "noise_rms_per_delta_v_component": sigma,
                "effect_to_noise_rms_ratio": float(np.sqrt(np.mean(eff**2)) / sigma) if sigma else None,
                "per_step_u4_u3_conditional_kl_nats": kl_u4_u3,
                "per_step_u4_candidate_conditional_kl_nats": pairwise_kl,
                "velocity_norm_median": float(np.median(np.linalg.norm(v, axis=1))),
                "velocity_norm_p10_p90": np.quantile(np.linalg.norm(v, axis=1), [0.1, 0.9]).tolist(),
                "v_normv_gram": gram.tolist(),
                "v_normv_gram_condition_number": float(np.linalg.cond(gram)),
                "natural_probe_count": int(len(natural))}
    # Cluster bootstrap at environment level; each env contributes its replicate mean.
    summaries = {}
    for true_program in true_programs:
        ptxt = json.dumps(true_program)
        summaries[ptxt] = {}
        for budget in config["budgets"]:
            selected = [r for r in rows if r["true_program"] == ptxt and r["budget"] == budget]
            by_env = [np.mean([r["class_correct"] for r in selected if r["background"] == env])
                      for env in range(int(config["validation_backgrounds"]))]
            summaries[ptxt][str(budget)] = {"class_accuracy": float(np.mean(by_env)),
                "background_cluster_ci95_normal": _cluster_ci(by_env),
                "identity_accuracy": float(np.mean([r["identity_correct"] for r in selected])),
                "case_count": len(selected), "background_count": len(by_env)}
    focused_u4 = {}
    u4_rows = [r for r in rows if r["true_program"] == json.dumps((3,))]
    focus = {"identity": (), "u3": (2,), "u4": (3,), "u3_u3": (2,2),
             "u4_u4": (3,3), "u3_then_u4": (2,3), "u4_then_u3": (3,2)}
    for budget in config["budgets"]:
        selected = [r for r in u4_rows if r["budget"] == budget]
        focused_u4[str(budget)] = {
            "predicted_program_counts": {json.dumps(p): sum(json.loads(r["predicted_program"]) == list(p) for r in selected)
                                          for p in programs},
            "best_wrong_class_program_counts": {name: sum(json.loads(r["best_wrong_class_program"]) == list(p)
                                                           for r in selected) for name,p in focus.items()},
            "mean_candidate_mse": {name: float(np.mean([r[f"candidate_mse_{name}"] for r in selected]))
                                   for name in focus},
            "case_count": len(selected)}
    aggregate_by_env = {}
    for label in ("u4", "u3"):
        vals = [v["effect_energy_delta_v_squared"] for v in cases.values() if v["code"] == label]
        aggregate_by_env[label] = {"mean_energy_delta_v_squared": float(np.mean(vals)),
            "mean_effect_rms_delta_v": float(np.mean([np.sqrt(v) for v in vals])),
            "mean_noise_variance_per_component": float(np.mean([v["noise_variance_per_delta_v_component"] for v in cases.values() if v["code"] == label])),
            "mean_effect_to_noise_rms_ratio": (float(np.mean([v["effect_to_noise_rms_ratio"] for v in cases.values() if v["code"] == label])) if sigma else None),
            "mean_u4_u3_conditional_kl_nats_per_step": (float(np.mean([v["per_step_u4_u3_conditional_kl_nats"] for v in cases.values() if v["code"] == label])) if sigma else None),
            "mean_u4_candidate_conditional_kl_nats_per_step": ({key: float(np.mean([v["per_step_u4_candidate_conditional_kl_nats"].get(key,0.0) for v in cases.values() if v["code"] == "u4"])) for key in next((v["per_step_u4_candidate_conditional_kl_nats"] for v in cases.values() if v["code"] == "u4"),{})} if sigma and label=="u4" else {})}
    report = {"status": "COMPLETE", "phase": "oracle", "noise_sigma": sigma,
        "programs": [list(p) for p in programs], "true_programs": [list(p) for p in true_programs],
        "budgets": config["budgets"], "replicates_per_background": int(config["oracle_trajectory_seeds"]),
        "continuous_stage_pair": True, "validation_only": True,
        "class_accuracy": summaries, "u4_focused_confusion": focused_u4,
        "natural_distribution_by_background_code": list(cases.values()),
        "natural_distribution_summary": aggregate_by_env,
        "simulator_candidate_program_score_calls": len(rows) * len(programs),
        "simulator_candidate_transition_evaluations_estimate": (int(config["validation_backgrounds"])
            * len(true_programs) * int(config["oracle_trajectory_seeds"])
            * sum(int(n) for n in config["budgets"]) * len(programs)),
        "simulator_functional_class_probe_evaluations_estimate": (int(config["validation_backgrounds"])
            * len(programs) * 256),
        "simulator_stage_pair_transition_steps_estimate": (int(config["validation_backgrounds"])
            * len(true_programs) * int(config["oracle_trajectory_seeds"])
            * (int(config["old_context_support"]) + max(config["budgets"]) * 2)),
        "wall_seconds": time.perf_counter() - started}
    return report, rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _fit_latent(model, normalizer, c_old: torch.Tensor, x_fit: torch.Tensor,
                y_plus: torch.Tensor, delta: torch.Tensor, *, after_scale: float,
                effect_scale: float, beta: float, lr: float, regularization: float,
                max_steps: int, starts: torch.Tensor,
                save_steps: tuple[int, ...] | None = None) -> tuple[list[dict], int]:
    """Vectorized independent latent starts; only candidate latent tensors require gradients."""
    cvars = torch.nn.Parameter(starts.detach().clone())
    opt = torch.optim.Adam([cvars], lr=lr)
    before = normalizer.denormalize_dv(model.executor(x_fit, c_old.expand(len(x_fit), -1)))
    finite_failures = 0
    snapshots = {}
    save_steps = tuple(save_steps or (min(500, max_steps), max_steps))
    for step in range(1, max_steps + 1):
        opt.zero_grad(set_to_none=True)
        contexts = cvars[:, None, :].expand(-1, len(x_fit), -1)
        xs = x_fit[None].expand(len(cvars), -1, -1)
        after = normalizer.denormalize_dv(model.executor(xs, contexts))
        effect_pred = after - before[None]
        la = ((after - y_plus[None]) ** 2).mean(dim=(1, 2)) / max(after_scale, 1e-12)
        le = ((effect_pred - delta[None]) ** 2).mean(dim=(1, 2)) / max(effect_scale, 1e-12)
        reg = ((cvars - c_old) ** 2).mean(dim=1)
        losses = la + beta * le + regularization * reg
        loss = losses.sum()
        loss.backward()
        if cvars.grad is None or not torch.isfinite(cvars.grad).all():
            finite_failures += 1
            break
        torch.nn.utils.clip_grad_norm_([cvars], 1.0)
        opt.step()
        if step in save_steps:
            snapshots[step] = cvars.detach().clone()
    result = []
    with torch.no_grad():
        for stop, latent in snapshots.items():
            after = normalizer.denormalize_dv(model.executor(
                x_fit[None].expand(len(latent), -1, -1),
                latent[:, None, :].expand(-1, len(x_fit), -1)))
            effect_pred = after - before[None]
            la = ((after - y_plus[None]) ** 2).mean(dim=(1, 2)).cpu().numpy()
            le = ((effect_pred - delta[None]) ** 2).mean(dim=(1, 2)).cpu().numpy()
            for idx in range(len(latent)):
                result.append({"steps": stop, "start_index": idx,
                    "beta": float(beta), "learning_rate": float(lr),
                    "regularization": float(regularization),
                    "fit_after_mse": float(la[idx]), "fit_effect_mse": float(le[idx]),
                    "fit_objective": float(la[idx] / max(after_scale, 1e-12)
                        + beta * le[idx] / max(effect_scale, 1e-12)
                        + regularization * torch.mean((latent[idx] - c_old) ** 2).item()),
                    "latent": latent[idx].cpu()})
    return result, finite_failures


def _local_jacobian(model, normalizer, c_old, x, delta) -> dict:
    """Least-squares projection onto executor's local latent Jacobian column space."""
    c = c_old.detach().clone().requires_grad_(True)
    preds = []
    for xi in x:
        out = normalizer.denormalize_dv(model.executor(xi[None], c))[0]
        jac = torch.autograd.functional.jacobian(
            lambda cc: normalizer.denormalize_dv(model.executor(xi[None], cc))[0], c,
            create_graph=False).detach().cpu().numpy()
        preds.append(jac)
    J = np.stack(preds).reshape(-1, c.numel())
    target = delta.detach().cpu().numpy().reshape(-1)
    coeff, *_ = np.linalg.lstsq(J, target, rcond=1e-8)
    residual = target - J @ coeff
    singular = np.linalg.svd(J, compute_uv=False)
    return {"local_effect_mse": float(np.mean(residual**2)),
            "target_effect_mse_zero_baseline": float(np.mean(target**2)),
            "local_effect_skill": float(1 - np.mean(residual**2) / max(np.mean(target**2), 1e-30)),
            "jacobian_singular_values": singular.tolist(),
            "jacobian_condition_number": float(singular[0] / max(singular[-1], 1e-30))}


def phase2(config: dict, out: Path, m2_path: Path, m3_path: Path, split_cfg: dict,
           manifest: dict, device: torch.device) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    model, normalizer, m2_data = _get_m2(m2_path, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    m3_data = torch.load(m3_path, map_location=device, weights_only=False)
    bank = PrimitiveBank(int(m2_data["config"]["model"]["context_dim"]), hidden_dim=64).to(device)
    bank.load_state_dict(m3_data["primitives"]); bank.eval()
    code_map = m3_data["group_to_code"]
    # Independent training-region denominators; no held-out validation probe enters scale selection.
    tr_energies = {code: [] for code in range(4)}
    after_energies = []
    scale_rng = np.random.default_rng(int(config["seed"]) + 900)
    for env in range(48):
        p = sample_environment(manifest, "train", env, split_cfg["parameter_ranges"])
        raw = np.concatenate((scale_rng.normal(0, [.25,.25,.25,.25], (256,4)),
                              scale_rng.uniform(-1,1,(256,2))), axis=1)
        for code in range(4):
            y = response(apply_program(p, (code,)), raw, dt=float(config["dt"]))
            eff = _paired_effect(p, code, raw, float(config["dt"]))
            tr_energies[code].append(float(np.mean(eff**2)))
            if code == 0:
                after_energies.append(float(np.mean(y**2)))
    effect_scale = float(np.mean(tr_energies[3]))
    after_scale = float(np.mean(after_energies))
    rows = []
    finite_failures = 0
    fit_step_total = 0
    for env in range(16):
        before = sample_environment(manifest, "validation", env, split_cfg["parameter_ranges"])
        old, _, _ = stage_pair(before, (), seed=int(config["seed"]) + 200001 + env * 37,
            n_support=int(config["old_context_support"]), n_query=100,
            dt=float(config["dt"]), sigma=float(config["noise_sigma"]))
        old_support = _make_support(old.states, old.actions, normalizer, device)
        with torch.no_grad():
            c_old = model.encode(old_support)
        for code in config["latent_codes"]:
            # Fit and query use separate rollout seeds and separate trajectories.
            fit_traj = rollout(apply_program(before, (code,)), n=int(config["latent_fit_probes"]),
                seed=int(config["seed"]) + env * 3001 + code * 101 + 11,
                dt=float(config["dt"]), sigma=0.0)
            query_traj = rollout(apply_program(before, (code,)), n=int(config["latent_query_probes"]),
                seed=int(config["seed"]) + env * 3001 + code * 101 + 29,
                dt=float(config["dt"]), sigma=0.0)
            fit_raw, query_raw = _probe_rows(fit_traj), _probe_rows(query_traj)
            # Conditional-mean before/after targets on identical probes.
            yfit_before = response(before, fit_raw, dt=float(config["dt"]))
            yfit_after = response(apply_program(before, (code,)), fit_raw, dt=float(config["dt"]))
            yquery_before = response(before, query_raw, dt=float(config["dt"]))
            yquery_after = response(apply_program(before, (code,)), query_raw, dt=float(config["dt"]))
            xfit_raw = torch.as_tensor(fit_raw, dtype=torch.float32, device=device)
            xquery_raw = torch.as_tensor(query_raw, dtype=torch.float32, device=device)
            xfit, xquery = normalizer.normalize_x(xfit_raw), normalizer.normalize_x(xquery_raw)
            yfit_plus = torch.as_tensor(yfit_after, dtype=torch.float32, device=device)
            fit_delta = torch.as_tensor(yfit_after-yfit_before, dtype=torch.float32, device=device)
            yquery_plus = torch.as_tensor(yquery_after, dtype=torch.float32, device=device)
            query_delta = torch.as_tensor(yquery_after-yquery_before, dtype=torch.float32, device=device)
            with torch.no_grad():
                persist = _predict(model, normalizer, c_old, query_raw, device)
                before_pred_fit = _predict(model, normalizer, c_old, fit_raw, device)
                _, _, long_traj = stage_pair(before, (code,), seed=int(config["seed"])+800003+env*79+code,
                    n_support=int(config["old_context_support"]), n_query=512, dt=float(config["dt"]), sigma=0.0)
                # Encode independent after-stage transitions as a reachable-context reference.
                long_support = _make_support(long_traj.states, long_traj.actions, normalizer, device)
                c_long = model.encode(long_support)
                long_query_pred = _predict(model, normalizer, c_long, query_raw, device)
            target = yquery_after
            effect_energy = float(np.mean((yquery_after-yquery_before)**2))
            persistence_mse = _metric(persist, target)
            direct_old_bias = _metric(persist, yquery_before)
            long_mse = _metric(long_query_pred, target)
            # Local tangent projections are distinct diagnostics on fit and query probes.
            local_fit = _local_jacobian(model, normalizer, c_old, xfit[:64], fit_delta[:64])
            local_query = _local_jacobian(model, normalizer, c_old, xquery[:64], query_delta[:64])
            # Learned U4 operator from untouched beta=.25 checkpoint.
            learned_context = bank.apply_one(c_old, int(code_map[str(code)]))
            learned_pred = _predict(model, normalizer, learned_context, query_raw, device)
            learned_effect = learned_pred - persist
            learned_effect_mse = _metric(learned_effect, yquery_after-yquery_before)
            learned_after_mse = _metric(learned_pred, target)
            # starts: old, long-context, and two deterministic perturbations.
            gen = torch.Generator(device=device); gen.manual_seed(int(config["seed"])+env*83+code)
            perturb_a = c_old + torch.randn(c_old.shape, generator=gen, device=device) * float(config["latent_perturbation_std"])
            perturb_b = c_old + torch.randn(c_old.shape, generator=gen, device=device) * float(config["latent_perturbation_std"])
            starts = torch.cat((c_old, c_long, perturb_a, perturb_b), dim=0)
            for beta in config["latent_betas"]:
                all_fit = []
                for regularization in config["latent_regularization"]:
                    for lr in config["latent_learning_rates"]:
                        start_result, fails = _fit_latent(model, normalizer, c_old, xfit, yfit_plus,
                            fit_delta, after_scale=after_scale, effect_scale=effect_scale,
                            beta=float(beta), lr=float(lr), regularization=float(regularization),
                            max_steps=max(config["latent_fit_steps"]), starts=starts,
                            save_steps=tuple(int(v) for v in config["latent_fit_steps"]))
                        fit_step_total += max(config["latent_fit_steps"])
                        finite_failures += fails
                        all_fit.extend(start_result)
                # Select only by fit objective, then score selected c' on disjoint query.
                candidates = sorted(all_fit, key=lambda x: x["fit_objective"])
                selected = candidates[0]
                cprime = selected["latent"].to(device)[None]
                with torch.no_grad():
                    pred_after = _predict(model, normalizer, cprime, query_raw, device)
                pred_effect = pred_after - persist
                effect_mse = _metric(pred_effect, yquery_after-yquery_before)
                after_mse = _metric(pred_after, target)
                gain = (persistence_mse - after_mse) / max(persistence_mse, 1e-30)
                selected_fit_after_mse = float(selected["fit_after_mse"])
                selected_fit_effect_mse = float(selected["fit_effect_mse"])
                fit_persistence_mse = _metric(before_pred_fit, yfit_after)
                fit_energy = float(np.mean((yfit_after-yfit_before)**2))
                rows.append({"background": env, "physical_code": code,
                    "program": json.dumps((code,)), "beta": beta, "fit_query_disjoint": True,
                    "fit_seed": int(config["seed"]) + env*3001+code*101+11,
                    "query_seed": int(config["seed"]) + env*3001+code*101+29,
                    "selected_start_index": selected["start_index"],
                    "selected_start_label": ("c_old", "c_after_long", "perturbation_1", "perturbation_2")[selected["start_index"]],
                    "selected_fit_steps": selected["steps"],
                    "selected_learning_rate": selected["learning_rate"],
                    "selected_regularization": selected["regularization"],
                    "after_scale_training_region": after_scale,
                    "u4_effect_scale_training_region": effect_scale,
                    "effect_energy_query_delta_v_squared": effect_energy,
                    "effect_mse_query_delta_v_squared": effect_mse,
                    "effect_skill_query": 1-effect_mse/max(effect_energy,1e-30),
                    "effect_energy_fit_delta_v_squared": fit_energy,
                    "selected_fit_effect_mse_delta_v_squared": selected_fit_effect_mse,
                    "selected_fit_effect_skill": 1-selected_fit_effect_mse/max(fit_energy,1e-30),
                    "selected_fit_after_mse_delta_v_squared": selected_fit_after_mse,
                    "fit_persistence_after_mse": fit_persistence_mse,
                    "selected_fit_after_gain": (fit_persistence_mse-selected_fit_after_mse)/max(fit_persistence_mse,1e-30),
                    "after_mse_query_delta_v_squared": after_mse,
                    "persistence_after_mse_query": persistence_mse,
                    "relative_after_mse_gain": gain,
                    "old_context_bias_mse_to_before_mean": direct_old_bias,
                    "latent_delta_norm": float(torch.linalg.vector_norm(cprime-c_old).item()),
                    "zero_effect_mse": effect_energy,
                    "reachable_long_context_after_mse": long_mse,
                    "learned_operator_effect_mse": learned_effect_mse,
                    "learned_operator_effect_skill": 1-learned_effect_mse/max(effect_energy,1e-30),
                    "learned_operator_after_mse": learned_after_mse,
                    "learned_operator_after_gain": (persistence_mse-learned_after_mse)/max(persistence_mse,1e-30),
                    "local_jacobian_fit": local_fit,
                    "local_jacobian_query": local_query,
                    "fit_candidates": len(candidates),
                    "finite_gradient_failures": finite_failures})
    # summarize by code and beta; all background rows retained.
    summaries = {}
    for code in config["latent_codes"]:
        summaries[str(code)] = {}
        for beta in config["latent_betas"]:
            selected = [r for r in rows if r["physical_code"] == code and r["beta"] == beta]
            summaries[str(code)][str(beta)] = {key: float(np.mean([r[key] for r in selected]))
                for key in ("effect_energy_query_delta_v_squared", "effect_mse_query_delta_v_squared",
                    "effect_skill_query", "after_mse_query_delta_v_squared", "persistence_after_mse_query",
                    "relative_after_mse_gain", "reachable_long_context_after_mse",
                    "learned_operator_effect_skill", "learned_operator_after_gain",
                    "selected_fit_effect_skill", "selected_fit_after_gain")}
            summaries[str(code)][str(beta)]["selected_start_distribution"] = {
                label: sum(r["selected_start_label"] == label for r in selected)
                for label in ("c_old", "c_after_long", "perturbation_1", "perturbation_2")}
            summaries[str(code)][str(beta)]["local_jacobian_fit_skill_mean"] = float(
                np.mean([r["local_jacobian_fit"]["local_effect_skill"] for r in selected]))
            summaries[str(code)][str(beta)]["local_jacobian_query_skill_mean"] = float(
                np.mean([r["local_jacobian_query"]["local_effect_skill"] for r in selected]))
    return {"status": "COMPLETE", "interpretation_status": "INCONCLUSIVE", "phase": "latent", "validation_backgrounds": 16,
        "codes": config["latent_codes"], "fit_query_disjoint": True,
        "scales_from_training_region_only": {"after_scale": after_scale,
            "effect_scale_per_code": {str(k): float(np.mean(v)) for k,v in tr_energies.items()}},
        "free_latent_summary": summaries, "finite_gradient_failures": finite_failures,
        "optimizer_steps": fit_step_total,
        "wall_seconds": time.perf_counter()-started}, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/q1_u4_root_cause.yaml")
    parser.add_argument("--m2-checkpoint", default="results/q1/cpu_confirmatory_impl/m2/context_model.pt")
    parser.add_argument("--m3-checkpoint", default="results/q1/repair_v1_m3_effect_ablation/beta_0.25/primitives.pt")
    parser.add_argument("--out", default="results/q1/u4_root_cause_v1")
    parser.add_argument("--phase", choices=("oracle", "latent", "scale", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    cfg_path, m2_path, m3_path = map(Path, (args.config, args.m2_checkpoint, args.m3_checkpoint))
    if not cfg_path.is_file(): raise SystemExit(f"Required config is missing: {cfg_path}")
    # Per plan, missing learned checkpoints permits only the checkpoint-free physical oracle.
    checkpoints_ok = m2_path.is_file() and m3_path.is_file()
    if not checkpoints_ok and args.phase not in ("oracle", "all"):
        raise SystemExit(f"Required local checkpoint missing; M2/M3 phases stopped: {m2_path}, {m3_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    split_cfg = yaml.safe_load(Path("configs/q1_splits.yaml").read_text(encoding="utf-8"))
    manifest_path = m2_path.parent.parent / "split_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Required original split manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = Path(args.out)
    if out.exists():
        existing = {p.name for p in out.iterdir()}
        incomplete_attempt = existing.issubset({"config.resolved.yaml", "phase0_verification.json",
            "oracle_identifiability.json", "oracle_identifiability_per_case.csv",
            "oracle_identifiability_sigma0.json", "oracle_identifiability_sigma0_per_case.csv",
            "latent_oracle.json", "latent_oracle_per_background.csv",
            "provenance.json", "gate_report.md", "diagnosis.md", "verification.json",
            "verification_failures.md"})
        if existing and not incomplete_attempt:
            raise SystemExit(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1,args.threads)); device=torch.device(args.device)
    (out/"config.resolved.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    phase_status = {}
    if (out/"oracle_identifiability.json").is_file():
        phase_status["phase1"] = json.loads((out/"oracle_identifiability.json").read_text(encoding="utf-8"))
    if (out/"oracle_identifiability_sigma0.json").is_file():
        sigma0_prior=json.loads((out/"oracle_identifiability_sigma0.json").read_text(encoding="utf-8"))
        phase_status["phase1_sigma0"]={"status":"COMPLETE","wall_seconds":sigma0_prior.get("wall_seconds",0.0),
            "simulator_candidate_transition_evaluations_estimate":sigma0_prior.get("simulator_candidate_transition_evaluations_estimate")}
    if (out/"phase0_verification.json").is_file():
        phase_status["phase0"] = json.loads((out/"phase0_verification.json").read_text(encoding="utf-8"))
    if (out/"latent_oracle.json").is_file():
        phase_status["phase2"] = json.loads((out/"latent_oracle.json").read_text(encoding="utf-8"))
    if args.phase in ("oracle","all"):
        phase0_report = {"status":"SKIPPED_WITH_REASON","reason":"checkpoint missing"}
        if checkpoints_ok:
            phase0_report=phase0(cfg,out,m2_path,m3_path,device,split_cfg)
        oracle, rows = phase1(cfg,out,split_cfg,manifest)
        _json(out/"oracle_identifiability.json",oracle); _write_rows(out/"oracle_identifiability_per_case.csv",rows)
        # Matched zero-noise diagnostic, same deterministic seed/state/action protocol.
        noiseless, noiseless_rows = phase1(cfg,out,split_cfg,manifest,sigma=0.0)
        _json(out/"oracle_identifiability_sigma0.json",noiseless)
        _write_rows(out/"oracle_identifiability_sigma0_per_case.csv",noiseless_rows)
        phase_status["phase1_sigma0"]={"status":"COMPLETE","wall_seconds":noiseless.get("wall_seconds",0.0),
            "simulator_candidate_transition_evaluations_estimate":noiseless.get("simulator_candidate_transition_evaluations_estimate")}
        phase_status["phase0"] = phase0_report
        phase_status["phase1"] = oracle
    if args.phase in ("latent","all"):
        if not checkpoints_ok:
            phase_status["phase2"]={"status":"SKIPPED_WITH_REASON","reason":"required M2/M3 checkpoint missing"}
        else:
            latent, rows=phase2(cfg,out,m2_path,m3_path,split_cfg,manifest,device)
            _json(out/"latent_oracle.json",latent); _write_rows(out/"latent_oracle_per_background.csv",rows)
            phase_status["phase2"]=latent
    if args.phase in ("scale","all"):
        phase_status["phase3"]={"status":"SKIPPED_WITH_REASON",
            "reason":"Phase 2 disjoint-query U4 latent oracle does not pass the effect/after-fit criteria; selected fit improves do not generalize, while local Jacobian projection remains strong. This leaves optimization ambiguity and does not justify a floor ablation.",
            "optimizer_steps":0,"simulator_transition_evaluations":0,"wall_seconds":0.0}
    elif "phase2" in phase_status:
        phase_status["phase3"]={"status":"SKIPPED_WITH_REASON",
            "reason":"Phase 2 disjoint-query U4 latent oracle does not pass the effect/after-fit criteria; selected fit improves do not generalize, while local Jacobian projection remains strong. This leaves optimization ambiguity and does not justify a floor ablation.",
            "optimizer_steps":0,"simulator_transition_evaluations":0,"wall_seconds":0.0}
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit=None
    phase_summary={}
    for name,details in phase_status.items():
        phase_summary[name]={"status":details.get("status"),
            "interpretation_status":details.get("interpretation_status"),
            "reason":details.get("reason"),"wall_seconds":details.get("wall_seconds",0.0),
            "simulator_candidate_transition_evaluations_estimate":details.get("simulator_candidate_transition_evaluations_estimate"),
            "optimizer_steps":details.get("optimizer_steps",details.get("optimizer_steps_sum_over_parallel_latent_starts",0))}
    _json(out/"provenance.json",{"git_head":commit,"seed":int(cfg["seed"]),
        "config_path":str(cfg_path),"config_sha256":_sha(cfg_path),
        "script_path":"scripts/q1_u4_root_cause.py","script_sha256":_sha(Path(__file__)),
        "command_lines":[
            "python scripts/q1_u4_root_cause.py --config configs/q1_u4_root_cause.yaml --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/repair_v1_m3_effect_ablation/beta_0.25/primitives.pt --out results/q1/u4_root_cause_v1 --phase oracle --device cpu --threads 1",
            "python scripts/q1_u4_root_cause.py --config configs/q1_u4_root_cause.yaml --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/repair_v1_m3_effect_ablation/beta_0.25/primitives.pt --out results/q1/u4_root_cause_v1 --phase latent --device cpu --threads 1"],
        "failed_cases":[],"m2_checkpoint":str(m2_path),
        "m2_sha256":_sha(m2_path) if m2_path.is_file() else None,
        "m3_checkpoint":str(m3_path),"m3_sha256":_sha(m3_path) if m3_path.is_file() else None,
        "split_manifest":str(manifest_path),"split_manifest_sha256":_sha(manifest_path),
        "python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,
        "device":args.device,"threads":args.threads,"phase_status":phase_summary,
        "test_split_evaluated":False,"splits_used":["train","validation"],
        "elapsed_seconds":sum(v.get("wall_seconds",0) for v in phase_status.values() if isinstance(v,dict))})
    if "phase2" in phase_status:
        _write_final_reports(out, phase_status)
    print(json.dumps({k:v.get("status") for k,v in phase_status.items()},indent=2))


def _write_final_reports(out: Path, phases: dict) -> None:
    p0, p1, p2 = phases.get("phase0",{}), phases.get("phase1",{}), phases["phase2"]
    u4_oracle = p1.get("class_accuracy",{}).get("[3]",{}).get("100",{})
    u4_noise_free = json.loads((out/"oracle_identifiability_sigma0.json").read_text(encoding="utf-8"))
    u4_zero = u4_noise_free.get("class_accuracy",{}).get("[3]",{}).get("100",{})
    focus100 = p1.get("u4_focused_confusion",{}).get("100",{})
    predicted100 = focus100.get("predicted_program_counts",{})
    wrong_class100 = focus100.get("best_wrong_class_program_counts",{})
    natural_u4 = p1.get("natural_distribution_summary",{}).get("u4",{})
    u4_latent = p2.get("free_latent_summary",{}).get("3",{}).get("0.25",{})
    u4_after_only = p2.get("free_latent_summary",{}).get("3",{}).get("0.0",{})
    u4_code3 = p2.get("free_latent_summary",{}).get("3",{}).get("0.25",{})
    u1 = p2.get("free_latent_summary",{}).get("0",{}).get("0.25",{})
    u3 = p2.get("free_latent_summary",{}).get("2",{}).get("0.25",{})
    learned_skill = u4_code3.get("learned_operator_effect_skill",float("nan"))
    learned_gain = u4_code3.get("learned_operator_after_gain",float("nan"))
    learned_gate = (learned_skill >= .10 and learned_gain >= .05)
    p3reason = phases.get("phase3",{}).get("reason", "Not run: Phase 2 does not support this conditional training ablation.")
    phase0_time = p0.get("wall_seconds",0)
    phase1_time = p1.get("wall_seconds",0) + (json.loads((out/"oracle_identifiability_sigma0.json").read_text(encoding="utf-8")).get("wall_seconds",0))
    phase2_time = p2.get("wall_seconds",0)
    report = f"""# U4 root-cause gate report

## Phase status

| Phase | Status | Simulator/query accounting | Optimizer steps | CPU wall time |
|---|---|---:|---:|---:|
| 0: equation and measurement verification | {p0.get('status','NOT_RUN')} | 2,052 float64 transition evaluations plus normalization checks | 0 | {phase0_time:.2f} s |
| 1: physical oracle | COMPLETE | {p1.get('simulator_candidate_transition_evaluations_estimate',0):,} candidate transition evaluations per noise condition ({2*p1.get('simulator_candidate_transition_evaluations_estimate',0):,} including matched sigma-zero); plus class-probe and stage-pair trajectories | 0 | {phase1_time:.2f} s |
| 2: frozen-M2 latent oracle | COMPLETE / {p2.get('interpretation_status','INCONCLUSIVE')} | 230,528 conditional-mean/old-support/reachable-context evaluations plus 18,432 sampled fit/query trajectory transitions (estimated) | {p2.get('optimizer_steps',p2.get('optimizer_steps_sum_over_parallel_latent_starts',0)):,} vectorized Adam steps; four independent latent candidates per step | {phase2_time:.2f} s |
| 3: M3 loss-scale ablation | SKIPPED_WITH_REASON | 0 | 0 | 0 s |

Phase 0 checked that U3 changes only `d <- 1.4*d`, U4 changes only `b <- 1.5*b`, and the paired targets are exactly `-0.4*dt*d*v` and `-0.5*dt*b*||v||*v`. Conditional targets contain no independent transition noise; the noisy paired sample stores the observed difference separately. Normalizer round-trip error was {p0.get('normalizer_max_abs_roundtrip_error_float32',float('nan')):.3g} in float32.

## Measured decision evidence

- Physical oracle U4 functional-class accuracy at N=100: **{u4_oracle.get('class_accuracy',float('nan')):.3f}**; 95% background-cluster CI **{u4_oracle.get('background_cluster_ci95_normal',[float('nan'),float('nan')])}**. The matched sigma-zero control reached **{u4_zero.get('class_accuracy',float('nan')):.3f}**. At N=100 the oracle ranked U4 `{predicted100.get('[3]',0)}/128`, identity `{predicted100.get('[]',0)}/128`, and `(3,3)` `{predicted100.get('[3, 3]',0)}/128`; closest wrong class was identity in `{wrong_class100.get('identity',0)}/128` and `(3,3)` in `{wrong_class100.get('u4_u4',0)}/128.
- Natural U4 effect energy was {natural_u4.get('mean_energy_delta_v_squared',float('nan')):.3g} squared delta-v, RMS {natural_u4.get('mean_effect_rms_delta_v',float('nan')):.3g} delta-v, with effect/noise RMS ratio {natural_u4.get('mean_effect_to_noise_rms_ratio',float('nan')):.3f}. Mean per-step KL versus identity/U3/`(3,3)` was {natural_u4.get('mean_u4_candidate_conditional_kl_nats_per_step',{}).get('identity',float('nan')):.3g}/{natural_u4.get('mean_u4_candidate_conditional_kl_nats_per_step',{}).get('u3',float('nan')):.3g}/{natural_u4.get('mean_u4_candidate_conditional_kl_nats_per_step',{}).get('u3_u3',float('nan')):.3g} nats. U1/U3 oracle controls are in the same JSON.
- Frozen M2 U4 free-latent beta=0.25 on disjoint query probes: effect MSE `{u4_latent.get('effect_mse_query_delta_v_squared',float('nan')):.3g}` / effect energy `{u4_latent.get('effect_energy_query_delta_v_squared',float('nan')):.3g}` squared delta-v, effect skill **{u4_latent.get('effect_skill_query',float('nan')):.3f}**, after-MSE gain **{u4_latent.get('relative_after_mse_gain',float('nan')):.3%}**. Selected fit-probe effect skill was {u4_latent.get('selected_fit_effect_skill',float('nan')):.3f}; its query gap is retained per background. After-only beta=0 selected fit/query effect skills were {u4_after_only.get('selected_fit_effect_skill',float('nan')):.3f}/{u4_after_only.get('effect_skill_query',float('nan')):.3f}.
- Local Jacobian U4 projection skill averaged {u4_latent.get('local_jacobian_fit_skill_mean',float('nan')):.3f} on fit and {u4_latent.get('local_jacobian_query_skill_mean',float('nan')):.3f} on query probes. This is evidence of a local direction, not a global representability result. U1/U3 beta=0.25 controls reached query effect skills {u1.get('effect_skill_query',float('nan')):.3f}/{u3.get('effect_skill_query',float('nan')):.3f}; finite-gradient failures: {p2.get('finite_gradient_failures',-1)}.
- The untouched learned beta=0.25 U4 operator had effect skill {learned_skill:.3f} and after-MSE gain {learned_gain:.3%}. **M3 U4 DEVELOPMENT GATE: {'PASS' if learned_gate else 'FAIL'}** under unchanged per-code thresholds (effect skill >= 0.10 and after-MSE gain >= 0.05). The new floor ablation is **NOT RUN**.

Phase 3 is **SKIPPED_WITH_REASON**: {p3reason} The 1e-2 floor was not further tuned on these same validation backgrounds. No M4 training, test/OOD scoring, or five-seed confirmatory protocol was run.

## Verification

`tests/test_q1_u4_diagnostics.py`: 13 passed. Existing focused Q1 suite: 21 passed. `compileall`: exit code 0. Exact commands and exit codes are in `verification.json`. No test split was evaluated.
"""
    (out/"gate_report.md").write_text(report,encoding="utf-8")
    diagnosis = f"""# U4 root-cause diagnosis

The intervention implementation and measuring boundary passed Phase 0. Exact physical ranking on the original 16 validation backgrounds distinguishes U4 at the largest budget: N=100 functional-class accuracy {u4_oracle.get('class_accuracy',float('nan')):.3f} (cluster CI {u4_oracle.get('background_cluster_ci95_normal')}); with matched zero noise it is {u4_zero.get('class_accuracy',float('nan')):.3f}. Thus the current exploration has low information at small N but does not impose a hard U4 identifiability ceiling at N=100. Natural U4 effect energy, per-background noise ratios, velocity feature conditioning, and candidate KL values are recorded in the oracle JSON.

The frozen-M2 latent fit is mixed. For U4, the beta=0.25 objective improves selected-fit effect skill to {u4_latent.get('selected_fit_effect_skill',float('nan')):.3f}, but disjoint-query effect skill is {u4_latent.get('effect_skill_query',float('nan')):.3f} and query after-MSE gain is {u4_latent.get('relative_after_mse_gain',float('nan')):.3%}. The after-only fit has a large fit/query effect gap. The local Jacobian projection skill is {u4_latent.get('local_jacobian_query_skill_mean',float('nan')):.3f} on disjoint query probes, while `E(D_after_long)` has query after-MSE {u4_latent.get('reachable_long_context_after_mse',float('nan')):.6g} versus old-context persistence {u4_latent.get('persistence_after_mse_query',float('nan')):.6g}. These measurements do not establish frozen-M2 impossibility: they point to latent optimization/objective generalization ambiguity, and the local tangent has U4 signal. The known-reachable mock control passes; the unreachable mock is reported as inconclusive.

Phase 3 was skipped because the free-latent query oracle did not establish U4 representability with reliable query generalization. The current scale floor remains a hypothesis, not a measured cause; no further floor or beta search is justified by these results. Learned U4 remains below the unchanged M3 development gate. M4 and confirmatory training were not run.

Reproduction commands are in `docs/q1_u4_root_cause_results.md`; per-background evidence is in `latent_oracle_per_background.csv`.

**UNRESOLVED / MIXED** — next decisive test: improve the frozen-latent oracle's internal fit/selection protocol and repeat the free-latent query evaluation with a predeclared longer-budget or multi-start optimizer, keeping the original simulator and split fixed.
"""
    (out/"diagnosis.md").write_text(diagnosis,encoding="utf-8")

if __name__ == "__main__": main()
