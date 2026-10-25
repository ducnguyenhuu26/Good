"""Fresh, energy-controlled pre-change excitation study for Q1 V5 (CPU).

The pre-change action schedules depend only on time and predeclared seeds. The
physical simulator is confined to generation and offline scoring. The frozen
V5 response and other operators are held fixed; only the quadratic-drag ratio
is re-estimated from opaque-label training transitions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import torch

from q1.datasets import rollout
from q1.exploration import action_at
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS, DynamicsParams, apply_program, response as simulator_response, sample_params, transition
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import gaussian_posterior
from q1_v2.inference import composed as v2_composed, rank_programs as v2_rank
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import design as v2_design
from q1_v3.mechanisms import LegacyBoundedOddAction, mean_delta_v as v3_mean
from q1_v3.operators import MechanismOperatorBank
from q1_v3.program_search import predict_effect, score_program_budgets
from q1_v3.task_inference import fit_task_map
from q1.model import ContextDynamicsModel, TransitionNormalizer
from q1.primitives import PrimitiveBank
from q1.evaluate import _direct_context
from q1_v5.model import ProgramModel, fit_theta, normal_mixture_shrink, response as v5_response
from scripts.q1_v5_fixed_confirmatory import select_v3_candidate
from scripts.q1_v5_v3_corrected_reanalysis import v3_mean_query
from q1_v5.prechange_design import (
    ENERGY_TARGET, POLICIES, action_energy, action_sequence, estimate_multiplicative_ratio,
    functional_class, matched_population_b, matched_population_u4_effect,
    prechange_old_stage_information, stage_corrected_jacobian,
)
from q1_v5.identifiability import efficient_information

DT, SIGMA = .05, .0005
BUDGETS = (5, 10, 20, 50, 100)
TRAIN_CAL_SEED = 20261701
DEV_SEED = 20261733
NTRAIN, NDEV = 40, 32
FROZEN_PATH = ROOT / "results/q1/q1_v5_fixed_confirmatory_dev/training_records.json"
OUT = ROOT / "results/q1/q1_v5_prechange_confirmatory_dev"
EXPLORATORY_SEEDS = {20261105, 20261229, 20261347, 20261403, 440010}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _boot(values, seed: int, reps: int = 5000) -> list[float | None]:
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    sample = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]


def _by_background(rows: list[dict], field: str, where=lambda row: True) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if where(row):
            grouped.setdefault(str(row["background"]), []).append(float(row[field]))
    return {key: float(np.mean(value)) for key, value in grouped.items()}


def _paired(a: dict[str, float], b: dict[str, float], seed: int) -> dict:
    keys = sorted(set(a) & set(b)); delta = [a[k]-b[k] for k in keys]
    return {"n_backgrounds": len(delta), "mean_difference": float(np.mean(delta)) if delta else None,
            "ci95_background_bootstrap": _boot(delta, seed)}


def _gather_development_seeds() -> tuple[set[int], list[dict]]:
    """Read only development split manifests; skip protected/test/final paths."""
    forbidden = set(EXPLORATORY_SEEDS)
    sources = []
    base = ROOT / "results/q1"
    for path in sorted(base.rglob("split_manifest.json")):
        rel = path.relative_to(ROOT).as_posix().lower()
        if any(term in rel for term in ("protected", "final_test", "test_split", "heldout_test")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values: set[int] = set()
        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ("seeds", "train_seeds", "parameter_seeds", "development_parameter_seeds",
                               "development_case_seeds", "old_policy_seeds", "case_seeds"):
                        collect(value)
                    walk(value)
            elif isinstance(node, list):
                for item in node: walk(item)
        def collect(node):
            if isinstance(node, (int, float)) and float(node).is_integer(): values.add(int(node))
            elif isinstance(node, list):
                for item in node: collect(item)
            elif isinstance(node, dict):
                for key, item in node.items():
                    if key in ("seeds", "development_parameter_seeds", "development_case_seeds"): collect(item)
        walk(data)
        if values:
            forbidden.update(values)
            sources.append({"path": rel, "sha256": _sha(path), "unique_seed_count": len(values)})
    return forbidden, sources


def _draw_seed_groups(seed: int, forbidden: set[int], groups: dict[str, int]) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed); used = set(forbidden); out = {}
    for name, count in groups.items():
        values = []
        while len(values) < count:
            s = int(rng.integers(1, 2**31-1))
            if s not in used:
                used.add(s); values.append(s)
        out[name] = values
    return out


def _load_frozen():
    raw = FROZEN_PATH.read_bytes(); data = json.loads(raw)["v5_training"]
    if data["selected_deployable_rule"] != "old_plus_eb":
        raise ValueError("the frozen training/inner rule is not old_plus_eb")
    ops = {int(k): {kk: (np.asarray(vv, float) if kk == "L" else vv) for kk, vv in v.items()}
           for k, v in data["operators"].items()}
    p2o = {int(k): int(v) for k, v in data["opaque_code_map_for_evaluator"].items()}
    qcodes = [k for k, op in ops.items() if op["mechanism"] == "quadratic_drag"]
    if len(qcodes) != 1:
        raise ValueError("frozen learned map must identify exactly one quadratic-drag group")
    model = ProgramModel(data["shared_action_response"], ops,
                         np.asarray(data["mean"], float), np.asarray(data["sd"], float))
    return model, data, p2o, qcodes[0], hashlib.sha256(raw).hexdigest()


def _rollout_actions(params, actions, *, initial_seed: int, noise_seed: int, initial=None):
    actions = np.asarray(actions, dtype=np.float64)
    init_rng = np.random.default_rng(initial_seed); noise_rng = np.random.default_rng(noise_seed)
    state = (init_rng.normal(0., [.25, .25, .25, .25]) if initial is None else np.asarray(initial).copy())
    states = np.empty((len(actions)+1, 4)); states[0] = state
    for t, action in enumerate(actions):
        state = transition(state, action, params, dt=DT, sigma=SIGMA, rng=noise_rng)
        if not np.isfinite(state).all() or np.linalg.norm(state) > 100:
            raise FloatingPointError(f"trajectory exceeded safety envelope at step {t}")
        states[t+1] = state
    x = np.c_[states[:-1], actions]
    return x, states[1:, 2:]-states[:-1, 2:], states[-1].copy(), states


def _post_rollout(params, program, end_state, seed: int, query_seed: int):
    ar = np.random.default_rng(seed + 104730); nr = np.random.default_rng(seed + 209760)
    actions = np.asarray([action_at(t, ar) for t in range(100)], dtype=np.float64)
    x, y, end, states = _rollout_actions(apply_program(params, program), actions,
                                          initial_seed=seed, noise_seed=seed+209760,
                                          initial=end_state)
    query = rollout(apply_program(params, program), n=256, seed=query_seed, dt=DT, sigma=SIGMA)
    xq = np.c_[query.states[:-1], query.actions]
    return x, y, xq, query.delta_v, actions, states


def _train_recalibration(frozen: ProgramModel, opaque_u4: int,
                         train_seeds: list[int], train_action_seeds: list[int],
                         train_noise_seeds: list[int]) -> tuple[ProgramModel, dict, list[dict]]:
    fits = []; diagnostics = []
    ratios = []; features = []; residuals = []
    for i, (pseed, aseed, nseed) in enumerate(zip(train_seeds, train_action_seeds, train_noise_seeds)):
        params = sample_params(np.random.default_rng(pseed), "train")
        actions = action_sequence("BLOCK_24_ENERGY_MATCHED", seed=aseed, n=512)
        x0, y0, _, _ = _rollout_actions(params, actions, initial_seed=aseed+31, noise_seed=nseed)
        # The physical U4 index is used only to generate this paired training observation.
        x1, y1, _, _ = _rollout_actions(apply_program(params, (3,)), actions,
                                          initial_seed=aseed+31, noise_seed=nseed)
        theta0, _, loss0 = fit_theta(x0, y0, frozen.c,
                                      prior=(frozen.prior_mean, frozen.prior_sd))
        theta1, _, loss1 = fit_theta(x1, y1, frozen.c, guess=theta0,
                                      prior=(frozen.prior_mean, frozen.prior_sd))
        fits.append(float(np.exp(theta0[5])))
        ratios.append(float(np.exp(theta1[5]-theta0[5])))
        null_b = theta0.copy(); null_b[5] = -30.
        baseline = v5_response(x1, null_b, frozen.c)
        v = x1[:, 2:4]
        feature = -DT * np.exp(theta0[5]) * np.linalg.norm(v, axis=1, keepdims=True) * v
        features.append(feature.reshape(-1)); residuals.append((y1-baseline).reshape(-1))
        diagnostics.append({"training_index": i, "parameter_seed": pseed,
                            "trajectory_seed": aseed, "noise_seed": nseed,
                            "old_fit_b": float(np.exp(theta0[5])),
                            "paired_fit_ratio_diagnostic": float(np.exp(theta1[5]-theta0[5])),
                            "old_fit_loss": loss0, "after_fit_loss": loss1})
    ratio = estimate_multiplicative_ratio(np.concatenate(features), np.concatenate(residuals))
    bcal = matched_population_b(np.asarray(fits))
    ops = {k: dict(v) for k, v in frozen.ops.items()}
    ops[opaque_u4] = dict(ops[opaque_u4], ratio=ratio)
    recalibrated = ProgramModel(frozen.c, ops, frozen.prior_mean, frozen.prior_sd)
    calibration = {**bcal, "operator_ratio_train_only": ratio,
                   "per_background_ratio_median_train_only": float(np.median(ratios)),
                   "training_background_count": len(train_seeds),
                   "calibration_source": "training paired noisy transitions only",
                   "operator_label_input": "opaque learned quadratic-drag group",
                   "ratio_formula": "1 + sum(feature * residual) / sum(feature^2)"}
    return recalibrated, calibration, diagnostics


def _load_v3():
    from scripts.q1_v5_fixed_confirmatory import load_baselines
    return load_baselines()["v3"]


def _load_v2():
    from scripts.q1_v5_fixed_confirmatory import load_baselines
    return load_baselines()["v2"]


def _v3_predictions(x0, y0, xa, ya, xq, budgets, baseline):
    action, bank, ck, code_map = baseline
    programs = [tuple(code_map[c] for c in pg) for pg in PROGRAMS]
    fit = fit_task_map(torch.as_tensor(x0, dtype=torch.float64), torch.as_tensor(y0, dtype=torch.float64),
                       action, ck["prior_mean"], ck["prior_covariance"], dt=DT, sigma=SIGMA, starts=1, maxiter=100)
    all_scores = score_program_budgets(torch.as_tensor(xa, dtype=torch.float64), torch.as_tensor(ya, dtype=torch.float64),
                                        fit.theta.detach(), fit.covariance.detach(), bank, action, programs,
                                        budgets, dt=DT, sigma=SIGMA)
    o2p = {v: k for k, v in code_map.items()}; out = {}
    for n in budgets:
        selected = select_v3_candidate(all_scores[n]); pred = tuple(o2p[c] for c in selected["program"])
        qpred = v3_mean_query(xq, selected["theta_mean"], action)
        out[n] = (pred, qpred, fit, selected)
    return out


def _v2_predictions(x0, y0, xa, ya, xq, budgets, baseline):
    basis, bank, ck, code_map = baseline
    programs = [tuple(code_map[c] for c in pg) for pg in PROGRAMS]
    post = gaussian_posterior(v2_design(basis, torch.as_tensor(x0, dtype=torch.float64)),
                              torch.as_tensor(y0, dtype=torch.float64), ck["prior_mean"],
                              ck["prior_covariance"], float(ck["sigma2"]))
    out = {}
    for n in budgets:
        ranked = v2_rank(bank, post, programs, v2_design(basis, torch.as_tensor(xa[:n], dtype=torch.float64)),
                         torch.as_tensor(ya[:n], dtype=torch.float64), float(ck["sigma2"]), 0.)
        selected = ranked[0]; opaque_program = selected[0]
        pred = tuple(next(k for k,v in code_map.items() if v == c) for c in opaque_program)
        mat, bias = v2_composed(bank, opaque_program)
        qpred = (v2_design(basis, torch.as_tensor(xq, dtype=torch.float64)) @ (mat @ post.mean + bias)).detach().numpy().reshape(-1,2)
        out[n] = (pred, qpred)
    return out


def _v5_prediction(model, x0, y0, xa, ya, xq, budgets, calibration, opaque_programs):
    theta, cov, _ = model.old_posterior(x0, y0)
    b, se = float(np.exp(theta[5])), float(np.exp(theta[5])*np.sqrt(max(cov[5,5], 0.)))
    b_eb, _ = normal_mixture_shrink(b, se, float(calibration["b_mean"]), float(calibration["b_sd"]))
    theta_eb = theta.copy(); theta_eb[5] = np.log(max(b_eb, 1e-9))
    pop_theta = theta_eb.copy(); pop_theta[5] = np.log(float(calibration["b_mean"]))
    legacy_theta = np.asarray(model.prior_mean, dtype=np.float64).copy()
    scored = model.score_candidates(xa, ya, theta_eb, cov, opaque_programs, budgets)
    o2p = None
    return theta, cov, theta_eb, float(b), float(b_eb), pop_theta, legacy_theta, scored


def _load_m4_context(old_states, old_actions, old_y, baseline):
    m2, normalizer, m2ck, m4, m4map, penalty = baseline
    support = np.c_[old_states[:-1], old_actions, old_y].astype(np.float32)
    st = torch.as_tensor(support, dtype=torch.float32)
    st = torch.cat((normalizer.normalize_x(st[:, :6]), normalizer.normalize_dv(st[:, 6:])), -1)[None]
    with torch.no_grad(): context = m2.encode(st)
    return context, m2, normalizer, m2ck, m4, m4map, penalty


def _v5_effect(model, xq, theta, opaque_program):
    return v5_response(xq, model.transform(theta, opaque_program), model.c)-v5_response(xq, theta, model.c)


def _collect_saved_background_seeds() -> tuple[set[int], list[dict]]:
    """Compatibility name retained for tests; never opens protected split paths."""
    return _gather_development_seeds()


def execute(args) -> None:
    global OUT
    OUT = Path(args.output)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refuse to overwrite nonempty output {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    frozen, records, p2o, opaque_u4, frozen_sha = _load_frozen()
    forbidden, audit_sources = _collect_saved_background_seeds()
    groups = {
        "train_parameter": args.train_backgrounds,
        "train_trajectory": args.train_backgrounds,
        "train_noise": args.train_backgrounds,
        "development_idlike_parameter": args.backgrounds,
        "development_ood_parameter": args.backgrounds,
        "development_old_initial": 2*args.backgrounds,
        "development_old_noise": 2*args.backgrounds,
        "development_old_action": len(POLICIES)*2*args.backgrounds,
        "development_case": 2*args.backgrounds*len(PROGRAMS),
        "development_query": 2*args.backgrounds*len(PROGRAMS),
    }
    train_groups={k:v for k,v in groups.items() if k.startswith("train_")}
    dev_groups={k:v for k,v in groups.items() if k.startswith("development_")}
    train_seeds = _draw_seed_groups(args.training_seed, forbidden, train_groups)
    forbidden.update(s for values in train_seeds.values() for s in values)
    dev_seeds = _draw_seed_groups(args.seed, forbidden, dev_groups)
    seeds = {**train_seeds, **dev_seeds}
    trp = seeds["train_parameter"]
    recal, calibration, train_rows = _train_recalibration(
        frozen, opaque_u4, trp, seeds["train_trajectory"], seeds["train_noise"])
    opaque_programs = [tuple(p2o[c] for c in program) for program in PROGRAMS]
    o2p = {v:k for k,v in p2o.items()}

    # The baseline checkpoints are only used in direct, matched-sequence cells.
    v2base = _load_v2(); v3base = _load_v3()
    # Load M4/its frozen M2 executor from the same baseline bundle.
    from scripts.q1_v5_fixed_confirmatory import load_baselines
    b = load_baselines(); m2, normalizer, m2ck = b["m2"]; m4, m4map, m4pen = b["m4"]
    m4base = (m2, normalizer, m2ck, m4, m4map, m4pen)

    config = {
        "protocol": "Q1_PRECHANGE_EXCITATION_CONFIRMATORY_DEVELOPMENT",
        "protocol_status": "new separate development protocol; old fixed and post-change active FAIL verdicts immutable",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "master_seed": args.seed, "training_seed": args.training_seed,
        "n_train": args.train_backgrounds, "n_development_per_regime": args.backgrounds,
        "dt": DT, "sigma": SIGMA, "N_old": 96, "N_adapt": 100, "N_query": 256,
        "budgets": list(BUDGETS), "programs": [list(p) for p in PROGRAMS],
        "estimator_rule": "frozen old_plus_eb; same train-only linear-b calibration for learner and population comparator",
        "recalibration_scope": "U4 quadratic-drag operator ratio only; all other fitted response/operators frozen",
        "policies": list(POLICIES), "energy_target_matched_policies": ENERGY_TARGET,
        "post_change_policy": "original sinusoidal-with-jitter, unchanged",
        "action_bounds": [-1., 1.],
        "predeclared_primary_comparisons": [
            "energy-matched block-24 vs random-corner and multi-frequency-smooth I_eff(log b), both regimes",
            "U4 oracle-code effect skill vs train-calibrated population, paired lower 95% CI > 0, both regimes",
            "U1-U3 effect skill and ordered-pair/class recovery noninferior to original sinusoidal by 0.02",
        ],
        "program_selection": "score all 21 candidates; only canonicalize functional class after prediction",
        "protected_test_used": False,
        "source_training_records_sha256": frozen_sha,
    }
    split = {
        "master_seed": args.seed, "training_seed": args.training_seed,
        "sets": {k: v for k,v in seeds.items()},
        "saved_development_seed_sources": audit_sources,
        "exploratory_seeds_explicitly_excluded": sorted(EXPLORATORY_SEEDS),
        "all_seed_groups_pairwise_disjoint": len(set(sum(seeds.values(), []))) == sum(map(len, seeds.values())),
        "protected_test_used": False,
        "parameter_seeds": {}, "old_trajectory_seeds": {}, "case_seeds": {}, "query_seeds": {},
    }
    for region, pkey, off in (("IDLIKE", "development_idlike_parameter", 0),
                              ("BACKGROUND_OOD", "development_ood_parameter", args.backgrounds)):
        split["parameter_seeds"][region] = seeds[pkey]
        split["old_trajectory_seeds"][region] = {
            "initial": seeds["development_old_initial"][off:off+args.backgrounds],
            "noise": seeds["development_old_noise"][off:off+args.backgrounds],
            "by_background": {f"{region}-{bi:03d}": {
                policy: seeds["development_old_action"][((off+bi)*len(POLICIES)+pi)]
                for pi, policy in enumerate(POLICIES)} for bi in range(args.backgrounds)},
        }
        split["case_seeds"][region] = [seeds["development_case"][((off+bi)*len(PROGRAMS)+pi)]
                                         for bi in range(args.backgrounds) for pi in range(len(PROGRAMS))]
        split["query_seeds"][region] = [seeds["development_query"][((off+bi)*len(PROGRAMS)+pi)]
                                          for bi in range(args.backgrounds) for pi in range(len(PROGRAMS))]
    (OUT / "config.resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "split_manifest.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (OUT / "train_only_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (OUT / "training_and_inference_audit.json").write_text(json.dumps({
        "training_fits": train_rows,
        "training_only_calibration": calibration,
        "physical_identity_in_learner": False,
        "physical_identity_usage": "generator and offline evaluator only",
        "prior_mean_legacy_population_baseline_reported_secondary": True,
        "operator_ratio_fit_formula": "coefficient of post residual over b=0 baseline on learned old-b feature",
    }, indent=2), encoding="utf-8")

    cases=[]; effects=[]; energies=[]; fisher_rows=[]; b_errors=[]; confusion=[]; action_hashes=[]
    # Baselines are evaluated only on original sinusoidal and energy-matched block cells.
    baseline_policy_set = {"ORIGINAL_SINUSOIDAL", "BLOCK_24_ENERGY_MATCHED"}
    old_init = seeds["development_old_initial"]; old_noise = seeds["development_old_noise"]
    regions=(
        ("IDLIKE", "train", seeds["development_idlike_parameter"], 0),
        ("BACKGROUND_OOD", "test_background_ood", seeds["development_ood_parameter"], args.backgrounds),
    )
    for regime, region_name, param_seeds, offset in regions:
      for bi,pseed in enumerate(param_seeds):
        background=f"{regime}-{bi:03d}"
        params=sample_params(np.random.default_rng(pseed), region_name)
        classes,_=functional_classes(params,dt=DT)
        old_by_policy={}
        for pi,policy in enumerate(POLICIES):
            action_seed=seeds["development_old_action"][(offset+bi)*len(POLICIES)+pi]
            actions=action_sequence(policy,seed=action_seed,n=96)
            action_hashes.append({"regime":regime,"background":background,"stage":"old",
                "policy":policy,"program":"UNKNOWN_PRECHANGE","seed":action_seed,
                "sha256":hashlib.sha256(np.asarray(actions,dtype="<f8").tobytes()).hexdigest()})
            x0,y0,end,states=_rollout_actions(params,actions,
                initial_seed=old_init[offset+bi],noise_seed=old_noise[offset+bi])
            theta,cov,loss=fit_theta(x0,y0,frozen.c,prior=(frozen.prior_mean,frozen.prior_sd))
            bmap=float(np.exp(theta[5]));se=bmap*np.sqrt(max(float(cov[5,5]),0.))
            beb,_=normal_mixture_shrink(bmap,se,float(calibration["b_mean"]),float(calibration["b_sd"]))
            theta_eb=theta.copy();theta_eb[5]=np.log(max(beb,1e-9))
            old_by_policy[policy]=(actions,x0,y0,end,states,theta,cov,theta_eb,bmap,beb,loss)
            energy=action_energy(actions)
            energies.append({"regime":regime,"background":background,"policy":policy,
                **energy,"terminal_old_state":json.dumps(end.tolist()),
                "terminal_velocity_norm":float(np.linalg.norm(end[2:])),
                "old_velocity_rms":float(np.sqrt(np.mean(x0[:,2:4]**2))),
                "old_fit_b_map":bmap,"old_fit_b_eb":float(beb),"old_fit_loss":loss})
            info=prechange_old_stage_information(x0,theta_eb,frozen.c,sigma=SIGMA)
            fisher_rows.append({"regime":regime,"background":background,"policy":policy,"program":"PRECHANGE_UNKNOWN",
                "N_old":96,"I_eff_pre_logb":info["efficient_information"],"local_logb_se_pre":info["crlb_se"]})
            b_errors.append({"regime":regime,"background":background,"policy":policy,
                "b_true_offline":params.b,"b_hat_map":bmap,"b_hat_eb":float(beb),
                "abs_b_error_map":abs(bmap-params.b),"abs_b_error_eb":abs(float(beb)-params.b),
                "relative_b_error_eb":abs(float(beb)-params.b)/params.b})

        for pi,program in enumerate(PROGRAMS):
          case_idx=(offset+bi)*len(PROGRAMS)+pi
          case_seed=seeds["development_case"][case_idx];query_seed=seeds["development_query"][case_idx]
          query_params=apply_program(params,program)
          # Same post action/noise and independent query seeds across every old-stage policy.
          post_by_policy={}
          for policy in POLICIES:
            _,x0,y0,end,states,theta,cov,theta_eb,bmap,beb,_=old_by_policy[policy]
            # _post_rollout accepts params and a generator program; pass the original
            # environment and physical program only to this data-generation helper.
            x1,y1,xq,yq,post_actions,post_states=_post_rollout(params,program,end,case_seed,query_seed)
            post_by_policy[policy]=(x1,y1,xq,yq,post_actions,post_states)
            if policy == POLICIES[0]:
                action_hashes.append({"regime":regime,"background":background,"stage":"post",
                    "policy":"ORIGINAL_SINUSOIDAL","program":json.dumps(program),"seed":case_seed+104730,
                    "sha256":hashlib.sha256(np.asarray(post_actions,dtype="<f8").tobytes()).hexdigest()})
            # Selected estimators: learned-recalibrated V5 on all policies; frozen
            # V5 and corrected V2/V3/M4 only on sinusoidal and energy-matched block.
            variants=[("V5_RECALIBRATED",recal)]
            if policy in baseline_policy_set:
                variants.append(("V5_FROZEN",frozen))
            for estimator,model in variants:
                ranked=model.score_candidates(x1,y1,theta_eb,cov,opaque_programs,BUDGETS)
                pop_theta=theta_eb.copy();pop_theta[5]=np.log(float(calibration["b_mean"]))
                legacy_pop=np.asarray(model.prior_mean,dtype=np.float64).copy()
                for n in BUDGETS:
                    score,opaque,_point_score=ranked[n][0]
                    pred=tuple(o2p[c] for c in opaque)
                    pred_theta=model.transform(theta_eb,opaque)
                    qpred=v5_response(xq,pred_theta,model.c)
                    class_ok=int(functional_class(pred)==functional_class(program))
                    row={"regime":regime,"background":background,"policy":policy,"estimator":estimator,
                         "true_program":json.dumps(program),"predicted_program":json.dumps(pred),"N":n,
                         "raw_program_correct":int(pred==program),"functional_class_correct":class_ok,
                         "independent_query_mse":float(np.mean((qpred-yq)**2)),
                         "selected_marginal_score":float(score),"terminal_old_state":json.dumps(end.tolist())}
                    cases.append(row)
                    # Program-selected effect is explicitly separated from the
                    # oracle-code U1-U4 transfer diagnostic below.
                    true_delta=simulator_response(query_params,xq,dt=DT)-simulator_response(params,xq,dt=DT)
                    selected_delta=qpred-v5_response(xq,theta_eb,model.c)
                    selected_pop=v5_response(xq,model.transform(pop_theta,opaque),model.c)-v5_response(xq,pop_theta,model.c)
                    energy_true=float(np.mean(true_delta**2))
                    for method,delta in (("SELECTED_PROGRAM",selected_delta),
                                         ("MATCHED_TRAIN_POP_SELECTED_PROGRAM",selected_pop),
                                         ("ZERO_EFFECT",np.zeros_like(true_delta))):
                        mse=float(np.mean((delta-true_delta)**2))
                        effects.append({"regime":regime,"background":background,"policy":policy,
                            "estimator":estimator,"true_program":json.dumps(program),"predicted_program":json.dumps(pred),
                            "N":n,"effect_type":"selected_program","baseline":method,
                            "effect_skill":1.-mse/max(energy_true,1e-30),"effect_mse":mse,"true_effect_energy":energy_true})
                    # Stage-consistent offline FI: old theta for pre data; transformed
                    # theta and chain-rule Jacobian for the observed post stage.
                    opaque_true=tuple(p2o[c] for c in program)
                    joint_j=stage_corrected_jacobian(x0,x1[:n],theta_eb,opaque_true,model)
                    finfo=efficient_information(joint_j,5,sigma=SIGMA)
                    fisher_rows.append({"regime":regime,"background":background,"policy":policy,
                        "estimator":estimator,"program":json.dumps(program),"N_old":96,"N_post":n,
                        "I_eff_two_stage_logb":finfo["efficient_information"],
                        "local_logb_se_two_stage":finfo["crlb_se"],
                        "program_used_for_information":"offline evaluator only; not passed to action policy"})
                    if len(program)==1:
                        physical=program[0];opaque_code=p2o[physical]
                        actual=simulator_response(apply_program(params,program),xq,dt=DT)-simulator_response(params,xq,dt=DT)
                        learned=_v5_effect(model,xq,theta_eb,(opaque_code,))
                        pop_delta=matched_population_u4_effect(xq,theta_eb,
                            lambda xx,tt:v5_response(xx,tt,model.c),model.transform,
                            opaque_code,calibration)
                        legacy_delta=v5_response(xq,model.transform(legacy_pop,(opaque_code,)),model.c)-v5_response(xq,legacy_pop,model.c)
                        for label,delta in (("ORACLE_CODE_LEARNED",learned),
                                            ("MATCHED_TRAIN_POPULATION",pop_delta),
                                            ("LEGACY_FROZEN_LOG_MEAN_POPULATION",legacy_delta),
                                            ("ZERO_EFFECT",np.zeros_like(actual))):
                            mse=float(np.mean((delta-actual)**2));den=max(float(np.mean(actual**2)),1e-30)
                            effects.append({"regime":regime,"background":background,"policy":policy,
                                "estimator":estimator,"true_program":json.dumps(program),"predicted_program":json.dumps(pred),
                                "N":n,"effect_type":f"oracle_code_U{physical+1}","baseline":label,
                                "effect_skill":1.-mse/den,"effect_mse":mse,"true_effect_energy":den})
                    if program==(3,) and pred==(3,3) or program==(3,3) and pred==(3,):
                        confusion.append({"regime":regime,"background":background,"policy":policy,
                            "estimator":estimator,"true_program":json.dumps(program),
                            "predicted_program":json.dumps(pred),"N":n})

            if policy in baseline_policy_set:
                # Corrected V3 baseline, selected by marginal-score argmin.
                v3rank=_v3_predictions(x0,y0,x1,y1,xq,BUDGETS,v3base)
                for n,(pred,qpred,fit,selected) in v3rank.items():
                    cases.append({"regime":regime,"background":background,"policy":policy,"estimator":"V3_CORRECTED",
                        "true_program":json.dumps(program),"predicted_program":json.dumps(pred),"N":n,
                        "raw_program_correct":int(pred==program),
                        "functional_class_correct":int(functional_class(pred)==functional_class(program)),
                        "independent_query_mse":float(np.mean((qpred-yq)**2)),
                        "selected_marginal_score":float(selected["marginal_score"])})
                # Matched V2 checkpoint baseline.
                v2rank=_v2_predictions(x0,y0,x1,y1,xq,BUDGETS,v2base)
                for n,(pred,qpred) in v2rank.items():
                    cases.append({"regime":regime,"background":background,"policy":policy,"estimator":"V2_RETRAINED",
                        "true_program":json.dumps(program),"predicted_program":json.dumps(pred),"N":n,
                        "raw_program_correct":int(pred==program),
                        "functional_class_correct":int(functional_class(pred)==functional_class(program)),
                        "independent_query_mse":float(np.mean((qpred-yq)**2))})
                # M4 latent-context baseline with all 21 learned primitive sequences.
                ctx,m2,normalizer,m2ck,m4,m4map,m4pen=_load_m4_context(
                    old_by_policy[policy][4],old_by_policy[policy][0],old_by_policy[policy][2],m4base)
                mapped=[tuple(m4map[c] for c in pg) for pg in PROGRAMS]
                x1mat=np.c_[x1,y1].astype(np.float32); xq_t=torch.as_tensor(xq[:,:6],dtype=torch.float32)
                for n in BUDGETS:
                    tr=torch.as_tensor(x1mat[:n],dtype=torch.float32);xn=normalizer.normalize_x(tr[:,:6]);yn=normalizer.normalize_dv(tr[:,6:])
                    costs=[]
                    with torch.no_grad():
                        for pr in mapped:
                            latent=m4.apply_program(ctx,pr);pred_dv=normalizer.denormalize_dv(m2.executor(xn,latent))
                            nll=.5*(np.log(2*np.pi*float(m2ck["config"].get("predictive_variance",1e-4)))+
                                    (pred_dv-normalizer.denormalize_dv(yn)).square()/float(m2ck["config"].get("predictive_variance",1e-4)))
                            costs.append(float(nll.sum())+m4pen*len(pr))
                        idx=int(np.argmin(costs));pred=PROGRAMS[idx];latent=m4.apply_program(ctx,mapped[idx])
                        qpred=normalizer.denormalize_dv(m2.executor(normalizer.normalize_x(xq_t),latent)).numpy()
                    cases.append({"regime":regime,"background":background,"policy":policy,"estimator":"M4",
                        "true_program":json.dumps(program),"predicted_program":json.dumps(pred),"N":n,
                        "raw_program_correct":int(pred==program),"functional_class_correct":int(functional_class(pred)==functional_class(program)),
                        "independent_query_mse":float(np.mean((qpred-yq)**2))})
                if (pi+1) % 7 == 0:
                    print(f"{regime} bg {bi+1}/{args.backgrounds} program {pi+1}/{len(PROGRAMS)}",flush=True)

    _write_csv(OUT/"policy_action_energy.csv",energies)
    _write_csv(OUT/"action_sequence_hashes.csv",action_hashes)
    _write_csv(OUT/"fisher_and_b_error_per_background.csv",fisher_rows+b_errors)
    _write_csv(OUT/"per_background_per_program.csv",cases)
    _write_csv(OUT/"effect_per_background.csv",effects)
    _write_csv(OUT/"u4_double_intervention_confusions.csv",confusion)
    summary=_summarize(cases,effects,energies,fisher_rows,b_errors,confusion,args.seed)
    decision=_decide(summary)
    summary["decision"]=decision
    (OUT/"baseline_comparisons.json").write_text(json.dumps(summary["comparisons"],indent=2),encoding="utf-8")
    (OUT/"bootstrap_ci95.json").write_text(json.dumps(summary["metrics"],indent=2),encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"DECISION.md").write_text("# Q1 pre-change excitation development decision\n\n"+
        "New protocol only; original fixed-policy and post-change active verdicts remain FAIL.\n\n"+
        "```json\n"+json.dumps(decision,indent=2)+"\n```\n",encoding="utf-8")
    (OUT/"provenance.json").write_text(json.dumps({
        "branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        "start_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,
        "device":"cpu","elapsed_seconds":time.perf_counter()-started,
        "source_sha256":{str(Path(__file__).relative_to(ROOT)): _sha(Path(__file__)),
                         "q1_v5/prechange_design.py":_sha(ROOT/"q1_v5/prechange_design.py"),
                         "frozen_training_records.json":frozen_sha},
        "baseline_checkpoint_sha256":{
            "v2_operator":_sha(ROOT/"results/q1/q1_v2_bayesian_operators_dev1/operator_seed4201.pt"),
            "v3_primary":_sha(ROOT/"results/q1/q1_v3_mechanism_repair_dev1/v3_primary.pt"),
            "v3_operator":_sha(ROOT/"results/q1/q1_v3_mechanism_repair_dev1/operator_seed4201_sparse.pt"),
            "m4_primitives":_sha(ROOT/"results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt"),
        },
        "protected_test_used":False},indent=2),encoding="utf-8")
    print(json.dumps(decision,indent=2),flush=True)


def _summarize(cases,effects,energies,fisher_rows,b_errors,confusion,seed):
    metrics={};comparisons={}
    estimators=sorted(set(r["estimator"] for r in cases))
    for regime in ("IDLIKE","BACKGROUND_OOD"):
      for estimator in estimators:
       for policy in POLICIES:
        for n in BUDGETS:
            rows=[r for r in cases if r["regime"]==regime and r["estimator"]==estimator and r["policy"]==policy and r["N"]==n]
            if not rows: continue
            for key,field in (("functional_class_accuracy","functional_class_correct"),
                              ("raw_program_accuracy","raw_program_correct"),
                              ("independent_query_mse","independent_query_mse")):
                by=_by_background(rows,field)
                metrics[f"{regime}|{estimator}|{policy}|N={n}|{key}"]={
                    "mean":float(np.mean(list(by.values()))),"ci95_background":_boot(list(by.values()),seed+n+1),
                    "n_backgrounds":len(by)}
      for policy in POLICIES:
        er=[r for r in effects if r["regime"]==regime and r["policy"]==policy]
        for kind in ("oracle_code_U1","oracle_code_U2","oracle_code_U3","oracle_code_U4","selected_program"):
          for baseline in sorted(set(r["baseline"] for r in er if r["effect_type"]==kind)):
           for estimator in sorted(set(r["estimator"] for r in er if r["effect_type"]==kind)):
            for n in BUDGETS:
             cell=[r for r in er if r["effect_type"]==kind and r["baseline"]==baseline and r["estimator"]==estimator and r["N"]==n]
             if not cell: continue
             by=_by_background(cell,"effect_skill")
             metrics[f"{regime}|{policy}|{estimator}|{kind}|{baseline}|N={n}"]={
                 "mean_skill":float(np.mean(list(by.values()))),"ci95_background":_boot(list(by.values()),seed+n+51),
                 "n_backgrounds":len(by),"mean_absolute_effect_mse":float(np.mean([r["effect_mse"] for r in cell])),
                 "mean_true_effect_energy":float(np.mean([r["true_effect_energy"] for r in cell]))}
      for policy in POLICIES:
        fi=[r for r in fisher_rows if r["regime"]==regime and r["policy"]==policy]
        for field in ("I_eff_pre_logb","I_eff_two_stage_logb"):
            cell=[r for r in fi if field in r and (field!="I_eff_two_stage_logb" or r.get("N") == 100)]
            if cell:
                by=_by_background(cell,field)
                metrics[f"{regime}|{policy}|{field}"]={"mean":float(np.mean(list(by.values()))),
                    "ci95_background":_boot(list(by.values()),seed+67),"n_backgrounds":len(by)}
      for policy in POLICIES:
        b=[r for r in b_errors if r["regime"]==regime and r["policy"]==policy]
        if b:
            for field in ("abs_b_error_map","abs_b_error_eb","relative_b_error_eb"):
                by=_by_background(b,field)
                metrics[f"{regime}|{policy}|{field}"]={"mean":float(np.mean(list(by.values()))),
                    "ci95_background":_boot(list(by.values()),seed+71),"n_backgrounds":len(by)}
      # Primary paired policy comparisons are background-level averages.
      primary="V5_RECALIBRATED"
      for policy in POLICIES:
        a=_by_background([r for r in fisher_rows if r["regime"]==regime and r["policy"]==policy and "I_eff_pre_logb" in r],"I_eff_pre_logb")
        b=_by_background([r for r in fisher_rows if r["regime"]==regime and r["policy"]=="BLOCK_24_ENERGY_MATCHED" and "I_eff_pre_logb" in r],"I_eff_pre_logb")
        if a and b:
            comparisons[f"{regime}|{policy}_minus_block24|I_eff_pre"]=_paired(a,b,seed+81)
        for policy2 in ("RANDOM_CORNER_ENERGY_MATCHED","MULTIFREQ_SMOOTH_ENERGY_MATCHED","ORIGINAL_SINUSOIDAL",
                        "BLOCK_ROTATED_ENERGY_MATCHED","BLOCK_DWELL_12_ENERGY_MATCHED","BLOCK_DWELL_48_ENERGY_MATCHED"):
            a=_by_background([r for r in fisher_rows if r["regime"]==regime and r["policy"]=="BLOCK_24_ENERGY_MATCHED" and "I_eff_pre_logb" in r],"I_eff_pre_logb")
            b=_by_background([r for r in fisher_rows if r["regime"]==regime and r["policy"]==policy2 and "I_eff_pre_logb" in r],"I_eff_pre_logb")
            comparisons[f"{regime}|BLOCK_24_ENERGY_MATCHED_minus_{policy2}|I_eff_pre"]=_paired(a,b,seed+83)
        for estimator in (primary,"V5_FROZEN"):
            for code in (1,2,3):
                a=_by_background([r for r in effects if r["regime"]==regime and r["policy"]=="BLOCK_24_ENERGY_MATCHED" and r["estimator"]==estimator and r["effect_type"]==f"oracle_code_U{code}" and r["baseline"]=="ORACLE_CODE_LEARNED" and r["N"]==100],"effect_skill")
                b=_by_background([r for r in effects if r["regime"]==regime and r["policy"]=="ORIGINAL_SINUSOIDAL" and r["estimator"]==estimator and r["effect_type"]==f"oracle_code_U{code}" and r["baseline"]=="ORACLE_CODE_LEARNED" and r["N"]==100],"effect_skill")
                if a and b: comparisons[f"{regime}|BLOCK24_minus_sin|{estimator}|U{code}"]=_paired(a,b,seed+89+code)
        for est in ("V5_RECALIBRATED","V5_FROZEN"):
            a=_by_background([r for r in effects if r["regime"]==regime and r["policy"]=="BLOCK_24_ENERGY_MATCHED" and r["estimator"]==est and r["effect_type"]=="oracle_code_U4" and r["baseline"]=="ORACLE_CODE_LEARNED" and r["N"]==100],"effect_skill")
            b=_by_background([r for r in effects if r["regime"]==regime and r["policy"]=="BLOCK_24_ENERGY_MATCHED" and r["estimator"]==est and r["effect_type"]=="oracle_code_U4" and r["baseline"]=="MATCHED_TRAIN_POPULATION" and r["N"]==100],"effect_skill")
            if a and b: comparisons[f"{regime}|{est}|U4_minus_matched_train_population"]=_paired(a,b,seed+97)
        # Program accuracy/order comparisons and U4-vs-double confusion counts.
        for est in estimators:
            for pair in ((0,1),(1,0)):
                cell=[r for r in cases if r["regime"]==regime and r["policy"]=="BLOCK_24_ENERGY_MATCHED" and r["estimator"]==est and r["N"]==100 and tuple(json.loads(r["true_program"]))==pair]
                if cell:
                    by=_by_background(cell,"functional_class_correct")
                    metrics[f"{regime}|BLOCK_24_ENERGY_MATCHED|{est}|order={pair}|N100"]={"mean":float(np.mean(list(by.values()))),"ci95_background":_boot(list(by.values()),seed+103)}
    # Policy energy summaries are computed at background level; matched arms target 1.8.
    energy_summary={}
    for regime in ("IDLIKE","BACKGROUND_OOD"):
        for policy in POLICIES:
            cell=[r for r in energies if r["regime"]==regime and r["policy"]==policy]
            energy_summary[f"{regime}|{policy}"]={"mean_step_energy":float(np.mean([r["mean_squared_norm"] for r in cell])),
                "ci95_background":_boot([r["mean_squared_norm"] for r in cell],seed+111),
                "mean_terminal_velocity_norm":float(np.mean([r["terminal_velocity_norm"] for r in cell]))}
    return {"metrics":metrics,"comparisons":comparisons,"action_energy":energy_summary,
            "u4_double_confusions":confusion,"background_counts":{"IDLIKE":len(set(r["background"] for r in energies if r["regime"]=="IDLIKE")),
                "BACKGROUND_OOD":len(set(r["background"] for r in energies if r["regime"]=="BACKGROUND_OOD"))}}


def _decide(summary):
    cmp=summary["comparisons"];metric=summary["metrics"]
    counts=summary["background_counts"]
    info=all(cmp[f"{regime}|BLOCK_24_ENERGY_MATCHED_minus_{control}|I_eff_pre"]["ci95_background_bootstrap"][0]>0
             for regime in ("IDLIKE","BACKGROUND_OOD") for control in ("RANDOM_CORNER_ENERGY_MATCHED","MULTIFREQ_SMOOTH_ENERGY_MATCHED"))
    u4=all(cmp[f"{regime}|V5_RECALIBRATED|U4_minus_matched_train_population"]["ci95_background_bootstrap"][0]>0
           for regime in ("IDLIKE","BACKGROUND_OOD"))
    u123=all(cmp[f"{regime}|BLOCK24_minus_sin|V5_RECALIBRATED|U{code}"]["ci95_background_bootstrap"][0]>=-.02
             for regime in ("IDLIKE","BACKGROUND_OOD") for code in (1,2,3))
    order=all(metric[f"{regime}|BLOCK_24_ENERGY_MATCHED|V5_RECALIBRATED|order={pair}|N100"]["mean"]>=.8
              for regime in ("IDLIKE","BACKGROUND_OOD") for pair in ((0,1),(1,0)))
    v3present=all(any("|V3_CORRECTED|" in key and key.startswith(regime) for key in metric)
                  for regime in ("IDLIKE","BACKGROUND_OOD"))
    gates={"block_Ieff_beats_energy_matched_random_and_smooth_both_regimes":info,
           "U4_oracle_effect_beats_train_matched_population_both_regimes":u4,
           "U1_U3_effect_noninferior_to_sinusoidal_margin_0.02":u123,
           "ordered_pair_accuracy_at_least_0.80_both_regimes":order,
           "corrected_V3_baseline_present":v3present,
           "at_least_32_independent_backgrounds_per_regime":all(counts.get(r,0)>=32 for r in ("IDLIKE","BACKGROUND_OOD"))}
    adequate_n=gates["at_least_32_independent_backgrounds_per_regime"]
    return {"protocol":"Q1_PRECHANGE_EXCITATION_CONFIRMATORY_DEVELOPMENT",
        "verdict":"INCONCLUSIVE" if not adequate_n else ("Q1_PRECHANGE_PASS" if all(gates.values()) else "FAIL"),
        "status":"development-only; protected final test unopened",
        "gates":gates,"old_Q1_FIXED_POLICY_verdict":"FAIL (immutable)",
        "old_Q1_ACTIVE_EXCITATION_verdict":"FAIL (immutable)",
        "protected_test_used":False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=DEV_SEED)
    ap.add_argument("--training-seed",type=int,default=TRAIN_CAL_SEED)
    ap.add_argument("--backgrounds",type=int,default=NDEV)
    ap.add_argument("--train-backgrounds",type=int,default=NTRAIN)
    ap.add_argument("--output",default=str(OUT))
    execute(ap.parse_args())


if __name__=="__main__": main()
