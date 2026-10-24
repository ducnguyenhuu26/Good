"""Run the frozen Q1 v2 CPU development study using weak observed transitions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from q1.datasets import rollout, stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS, DynamicsParams, apply_program, sample_params, response
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import GaussianPosterior, gaussian_posterior, marginal_nll
from q1_v2.inference import rank_programs
from q1_v2.training import ObservedEpisode, design, fit_basis, fit_operator_bank


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(traj):
    return (np.c_[traj.states[:-1], traj.actions].astype(np.float32),
            traj.delta_v.astype(np.float32))


def episode(support, query, program=()):
    sx, sy = rows(support)
    qx, qy = rows(query)
    return ObservedEpisode(torch.from_numpy(sx), torch.from_numpy(sy),
                           torch.from_numpy(qx), torch.from_numpy(qy), tuple(program))


def collect_backgrounds(count, seed, forbidden, region, ranges):
    rng = np.random.default_rng(seed)
    seeds = []
    while len(seeds) < count:
        s = int(rng.integers(1, 2**31 - 1))
        if s not in forbidden and s not in seeds:
            seeds.append(s)
    params = [sample_params(np.random.default_rng(s), region, ranges) for s in seeds]
    return seeds, params


def gather_forbidden_seeds():
    values = set()
    sources = []
    for path in sorted((ROOT / "results/q1").rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        seeds = set()
        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"environment_seeds", "parameter_seeds", "saved_environment_parameter_seeds_for_collision_check_only"}:
                        if isinstance(value, dict):
                            for seq in value.values():
                                if isinstance(seq, list):
                                    seeds.update(int(x) for x in seq if isinstance(x, (int, float)))
                        elif isinstance(value, list):
                            seeds.update(int(x) for x in value if isinstance(x, (int, float)))
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(obj)
        if seeds:
            values.update(seeds)
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                            "unique_seed_count": len(seeds)})
    return values, sources


def tensor_input(x):
    return torch.as_tensor(x, dtype=torch.float32)


def make_training_data(train_params, inner_params, cfg, *, include_basis=True):
    fit_eps, inner_eps, op_eps, inner_op = [], [], [], []
    allowed = [tuple(p) for p in cfg["composition_training_pairs"]]
    for split_name, params_list, basis_out, op_out, base_seed in (
        ("fit", train_params, fit_eps, op_eps, 700000),
        ("inner", inner_params, inner_eps, inner_op, 900000)):
        for bidx, before in enumerate(params_list):
            if include_basis:
                for law_idx, program in enumerate([()] + [(i,) for i in range(4)] + allowed):
                    seed = base_seed + bidx * 1009 + law_idx * 37
                    law = apply_program(before, program)
                    support = rollout(law, n=cfg["old_support"], seed=seed,
                                      dt=cfg["dt"], sigma=cfg["sigma"])
                    query = rollout(law, n=32, seed=seed + 17, dt=cfg["dt"], sigma=cfg["sigma"])
                    basis_out.append(episode(support, query))
            for pidx, program in enumerate([(i,) for i in range(4)] + allowed):
                old, adapt, _ = stage_pair(before, program, seed=base_seed + bidx * 1009 + 500 + pidx * 31,
                    n_support=cfg["old_support"], n_query=cfg["adaptation_trajectory"],
                    dt=cfg["dt"], sigma=cfg["sigma"])
                ox, oy = rows(old)
                ax, ay = rows(adapt)
                # Training uses observed after-stage transition samples only.
                take = np.linspace(0, len(ax) - 1, 16, dtype=int)
                op_out.append(ObservedEpisode(torch.from_numpy(ox), torch.from_numpy(oy),
                    torch.from_numpy(ax[take]), torch.from_numpy(ay[take]), program))
    return fit_eps, inner_eps, op_eps, inner_op


def _scores_for_case(basis, bank, prior_mean, prior_cov, sigma2, old, adapt, programs, n,
                     length_penalty, point=False):
    oldx, oldy = rows(old)
    ax, ay = rows(adapt)
    oldpost = gaussian_posterior(design(basis, tensor_input(oldx)), tensor_input(oldy),
                                 prior_mean, prior_cov, sigma2)
    phi = design(basis, tensor_input(ax[:n]))
    y = tensor_input(ay[:n])
    ranked = rank_programs(bank, oldpost, programs, phi, y, sigma2, length_penalty,
                           point_estimate=point)
    return oldpost, ranked


def evaluate_cases(basis, banks, context_free_banks, prior_stats, params_by_regime, seeds_by_regime, cfg,
                   opaque_map, length_penalty, output, train_b_mean):
    basis.eval()
    for parameter in basis.parameters():
        parameter.requires_grad_(False)
    for bank, _ in banks.values():
        for parameter in bank.parameters():
            parameter.requires_grad_(False)
    for bank in context_free_banks.values():
        for parameter in bank.parameters():
            parameter.requires_grad_(False)
    background_rows, effect_rows = [], []
    rows_prog = list(PROGRAMS)
    for regime, backgrounds in params_by_regime.items():
        for bi, before in enumerate(backgrounds):
            print(f"Evaluating {regime} background {bi + 1}/{len(backgrounds)}", flush=True)
            bseed = int(seeds_by_regime[regime][bi])
            classes, _ = functional_classes(before, dt=cfg["dt"])
            for pi, true_program in enumerate(PROGRAMS):
                case_seed = bseed + 300000 + pi * 7919
                old, adapt, query = stage_pair(before, true_program, seed=case_seed,
                    n_support=cfg["old_support"], n_query=max(cfg["adaptation_trajectory"], cfg["query_trajectory"]),
                    dt=cfg["dt"], sigma=cfg["sigma"])
                ax, ay = rows(adapt)
                qx, qy = rows(query)
                qphi = design(basis, tensor_input(qx))
                target = tensor_input(qy).reshape(-1)
                for n in cfg["budgets"]:
                    method_results = {}
                    for seed, (bank, stat) in banks.items():
                        mean, cov, sigma2 = prior_stats[seed]
                        inverse_opaque = {v:int(k) for k,v in opaque_map.items()}
                        code_candidates = [tuple(opaque_map[str(c)] for c in p) for p in rows_prog]
                        oldpost, ranked = _scores_for_case(basis, bank, mean, cov, sigma2, old,
                            adapt, code_candidates, n, length_penalty)
                        predicted = tuple(inverse_opaque[c] for c in ranked[0][0])
                        cls_ok = classes[PROGRAMS.index(predicted)] == classes[pi]
                        # Assessment labels are accessed only after selection for MSE reporting.
                        a = ranked[0][0]
                        from q1_v2.inference import composed
                        am, bb = composed(bank, a)
                        pred_mean = am @ oldpost.mean + bb
                        pred = (qphi @ pred_mean).reshape(-1)
                        query_mse = float(torch.mean((pred - target) ** 2).detach())
                        _, point_ranked = _scores_for_case(basis, bank, mean, cov, sigma2, old,
                            adapt, code_candidates, n, length_penalty, point=True)
                        single_candidates = [()] + [(opaque_map[str(i)],) for i in range(4)]
                        _, single_ranked = _scores_for_case(basis, bank, mean, cov, sigma2, old,
                            adapt, single_candidates, n, length_penalty)
                        wrong_map = {str(i): opaque_map[str((i + 1) % 4)] for i in range(4)}
                        # Permute code meanings consistently at inference, keeping the learned bank fixed.
                        wrong_programs = [tuple(wrong_map[str(c)] for c in p) for p in rows_prog]
                        _, wrong_ranked = _scores_for_case(basis, bank, mean, cov, sigma2, old,
                            adapt, wrong_programs, n, length_penalty)
                        inverse_wrong = {v:int(k) for k,v in wrong_map.items()}
                        from q1_v2.inference import rank_programs as rank_v2
                        from q1_v2.bayes import GaussianPosterior as V2Posterior
                        prior_only = V2Posterior(mean, cov)
                        cfx, cfy = rows(adapt)
                        cf_ranked = rank_v2(context_free_banks[seed], prior_only, code_candidates,
                            design(basis, tensor_input(cfx[:n])), tensor_input(cfy[:n]), sigma2,
                            length_penalty)
                        # Offline paired-effect assessment on independent query probes.
                        effect_skill = None
                        analytical_u4_skill = None
                        context_swap_nll_increase = None
                        if len(true_program) == 1:
                            code = true_program[0]
                            from q1_v2.inference import composed as compose_effect
                            pa, pb = compose_effect(bank, (opaque_map[str(code)],))
                            predicted_effect = (qphi @ ((pa @ oldpost.mean + pb) - oldpost.mean)).reshape(-1, 2).detach().numpy()
                            probe_x = qx[:, :6].astype(np.float64)
                            true_effect = response(apply_program(before, true_program), probe_x, dt=cfg["dt"]) - response(before, probe_x, dt=cfg["dt"])
                            denom = float(np.mean(true_effect ** 2))
                            effect_skill = (1.0 - float(np.mean((predicted_effect - true_effect) ** 2)) / denom
                                            if denom > 0 else None)
                            # Swapping old support to a distinct background diagnoses dependence.
                            other = backgrounds[(bi + 1) % len(backgrounds)]
                            other_seed = int(seeds_by_regime[regime][(bi + 1) % len(backgrounds)]) + 300000 + pi * 7919
                            other_old = rollout(other, n=cfg["old_support"], seed=other_seed,
                                                dt=cfg["dt"], sigma=cfg["sigma"])
                            other_x, other_y = rows(other_old)
                            swapped_post = gaussian_posterior(design(basis, tensor_input(other_x)),
                                tensor_input(other_y), mean, cov, sigma2)
                            true_opaque = tuple(opaque_map[str(c)] for c in true_program)
                            swap_ranked = rank_v2(bank, swapped_post, [true_opaque],
                                design(basis, tensor_input(ax[:n])), tensor_input(ay[:n]), sigma2, length_penalty)
                            original_true_score = next(score for p, score, _ in ranked if p == true_opaque)
                            context_swap_nll_increase = float(swap_ranked[0][1] - original_true_score)
                            if code == 3:
                                v = probe_x[:, 2:4]
                                analytic_effect = -cfg["dt"] * 0.5 * float(train_b_mean) * np.linalg.norm(v,axis=1,keepdims=True) * v
                                analytical_u4_skill = (1.0 - float(np.mean((analytic_effect - true_effect) ** 2)) / denom
                                                       if denom > 0 else None)
                        method_results[seed] = {
                            "predicted": predicted, "class_correct": bool(cls_ok), "query_mse": query_mse,
                            "point_predicted": tuple(inverse_opaque[c] for c in point_ranked[0][0]),
                            "single_predicted": tuple(inverse_opaque[c] for c in single_ranked[0][0]),
                            "wrong_code_predicted": tuple(inverse_wrong[c] for c in wrong_ranked[0][0]),
                            "context_ablated_predicted": tuple(inverse_opaque[c] for c in cf_ranked[0][0]),
                            "best_wrong_class_minus_top_ranked_score": float(next((s for p, s, _ in ranked if
                                classes[PROGRAMS.index(tuple(inverse_opaque[c] for c in p))] != classes[pi]), ranked[-1][1]) - ranked[0][1]),
                            "old_posterior_trace": float(torch.trace(oldpost.covariance)),
                            "selected_covariance_trace": ranked[0][2]["posterior_trace"],
                            "effect_skill": effect_skill,
                            "analytical_u4_skill": analytical_u4_skill,
                            "context_swap_nll_increase": context_swap_nll_increase,
                            "identity_only_correct": classes[0] == classes[pi],
                        }
                    for seed, vals in method_results.items():
                        bank = banks[seed][0]
                        vals_single = [v for k, v in vals.items()]
                        true_code_program = tuple(opaque_map[str(c)] for c in true_program)
                        result = vals
                        background_rows.append({"regime": regime, "background_id": f"{regime}-{bi:03d}",
                            "background_seed": bseed, "optimizer_seed": seed,
                            "true_program": json.dumps(true_program), "N": n,
                            "predicted_program": json.dumps(result["predicted"]),
                            "functional_class_correct": int(result["class_correct"]),
                            "query_delta_v_mse": result["query_mse"],
                            "point_predicted_program": json.dumps(result["point_predicted"]),
                            "single_only_predicted_program": json.dumps(result["single_predicted"]),
                            "wrong_code_predicted_program": json.dumps(result["wrong_code_predicted"]),
                            "context_ablated_predicted_program": json.dumps(result["context_ablated_predicted"]),
                            "true_vs_best_wrong_class_margin": result["best_wrong_class_minus_top_ranked_score"],
                            "old_posterior_trace": result["old_posterior_trace"],
                            "selected_posterior_trace": result["selected_covariance_trace"]})
                        background_rows[-1]["effect_skill"] = result["effect_skill"]
                        background_rows[-1]["analytical_u4_constant_drag_skill"] = result["analytical_u4_skill"]
                        background_rows[-1]["context_swap_nll_increase"] = result["context_swap_nll_increase"]
                        background_rows[-1]["identity_only_correct"] = int(result["identity_only_correct"])
    return background_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/q1_v2.yaml")
    parser.add_argument("--run-dir", default="results/q1/q1_v2_bayesian_operators_dev1")
    parser.add_argument("--reuse-basis-checkpoint", default=None,
                        help="freeze an existing V2 basis/prior checkpoint while retraining operators")
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = ROOT / args.run_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite v2 artifacts: {out}")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(1)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    with (out / "config.resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Fresh split is frozen before data generation and fitting. Training and development seeds
    # are audited against every saved Q1 split/protocol manifest, including prior A/B/C backgrounds.
    forbidden, sources = gather_forbidden_seeds()
    train_seeds, train_params = collect_backgrounds(cfg["train_backgrounds"], cfg["seed"] + 1,
                                                    forbidden, "train", {"train": ((.75, 1.), (.28,.42), (.08,.14), (.18,.28))})
    forbidden.update(train_seeds)
    inner_seeds, inner_params = collect_backgrounds(cfg["inner_backgrounds"], cfg["seed"] + 2,
                                                    forbidden, "train", {"train": ((.75, 1.), (.28,.42), (.08,.14), (.18,.28))})
    forbidden.update(inner_seeds)
    id_seeds, id_params = collect_backgrounds(cfg["development_backgrounds_per_regime"], cfg["seed"] + 3,
                                               forbidden, "train", {"train": ((.75, 1.), (.28,.42), (.08,.14), (.18,.28))})
    forbidden.update(id_seeds)
    ood_ranges = {"dev_ood": {"B_diag": (1.08,1.2), "B_offdiag":(-.12,.12),
                              "d":(.49,.59), "b":(.16,.20), "kappa":(.32,.39)}}
    ood_seeds, ood_params = collect_backgrounds(cfg["development_backgrounds_per_regime"], cfg["seed"] + 4,
                                                forbidden, "dev_ood", ood_ranges)
    manifest = {"protocol":"fresh development only", "seed":cfg["seed"], "train":{},
        "inner_train":{}, "development_idlike":{}, "development_background_ood":{},
        "parameter_ranges": {"train": {"B_diag":[.75,1.0], "B_offdiag":[-.12,.12], "d":[.28,.42], "b":[.08,.14], "kappa":[.18,.28]},
          "dev_ood": {"B_diag":[1.08,1.2], "B_offdiag":[-.12,.12], "d":[.49,.59], "b":[.16,.20], "kappa":[.32,.39]}},
        "collision_audit_sources":sources, "test_splits_used":False}
    for key, seeds, params in [("train",train_seeds,train_params),("inner_train",inner_seeds,inner_params),
        ("development_idlike",id_seeds,id_params),("development_background_ood",ood_seeds,ood_params)]:
        manifest[key]={"seeds":seeds,"backgrounds":[{"B":p.B.tolist(),"d":p.d,"b":p.b,"kappa":p.kappa} for p in params]}
    (out/"split_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    # Hash split/provenance before model training starts.
    pretraining_provenance={"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        "config_sha256":sha256(config_path),"split_manifest_sha256":sha256(out/"split_manifest.json"),
        "python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
        "device":"cpu","train_backgrounds":len(train_params),"inner_backgrounds":len(inner_params),
        "development_test_split_used":False,"simulator_use":"data generation and evaluation oracles only"}
    (out/"provenance.pretraining.json").write_text(json.dumps(pretraining_provenance,indent=2),encoding="utf-8")

    fit_eps, inner_eps, op_eps, inner_op = make_training_data(
        train_params, inner_params, cfg, include_basis=not bool(args.reuse_basis_checkpoint))
    print(f"Generated weak observed episodes: basis={len(fit_eps)}/{len(inner_eps)} operator={len(op_eps)}/{len(inner_op)}", flush=True)
    basis_widths=tuple(cfg.get("basis_widths",(6,4,2)))
    basis = PartitionedBasis(widths=basis_widths,hidden=cfg["hidden"])
    # Noise variance is fixed from the declared train-only simulator calibration (sigma^2).
    sigma2 = float(cfg["sigma"] ** 2)
    if args.reuse_basis_checkpoint:
        basis_source=(ROOT/args.reuse_basis_checkpoint).resolve()
        saved=torch.load(basis_source,map_location="cpu",weights_only=False)
        basis.load_state_dict(saved["basis"])
        sigma2=float(saved["sigma2"])
        basis_stats={"prior_mean":saved["prior_mean"],"prior_covariance":saved["prior_covariance"],
                     "selected_epoch":saved.get("selected_epoch"),"curve":[]}
        for parameter in basis.parameters(): parameter.requires_grad_(False)
        torch.save(saved,out/"basis_pretrained.pt")
        (out/"basis_reuse.json").write_text(json.dumps({"source":basis_source.relative_to(ROOT).as_posix(),
            "source_sha256":sha256(basis_source),"basis_architecture":f"PartitionedBasis(widths={basis_widths}, hidden={cfg['hidden']})",
            "basis_and_prior_frozen":True},indent=2),encoding="utf-8")
        (out/"basis_training.json").write_text(json.dumps({"reused":True,"selected_epoch":basis_stats["selected_epoch"],
            "train_episode_count":0,"inner_episode_count":0,"weak_labels":"prior V2 checkpoint reused unchanged"},indent=2),encoding="utf-8")
        print("Reused and froze the original V2 basis/prior checkpoint",flush=True)
    else:
        basis_stats = fit_basis(basis, fit_eps, sigma2=sigma2, seed=cfg["seed"],
            epochs=cfg["basis_epochs"], batch_size=8, lr=cfg["learning_rate"], inner_episodes=inner_eps)
        print(f"Basis training complete; selected epoch {basis_stats['selected_epoch']}", flush=True)
        basis_cpu = {k:v.detach().cpu() for k,v in basis.state_dict().items()}
        torch.save({"basis":basis_cpu,"widths":basis_widths,"prior_mean":basis_stats["prior_mean"].cpu(),
                    "prior_covariance":basis_stats["prior_covariance"].cpu(),"sigma2":sigma2,
                    "selected_epoch":basis_stats["selected_epoch"]}, out/"basis_pretrained.pt")
        (out/"basis_training.json").write_text(json.dumps({"curve":basis_stats["curve"],
            "selected_epoch":basis_stats["selected_epoch"],"train_episode_count":len(fit_eps),
            "inner_episode_count":len(inner_eps),"weak_labels":"observed (s,a,delta_v), no simulator conditional means"},indent=2),encoding="utf-8")

    # Select one common regularizer on the fixed 8 inner training backgrounds, then refit both seeds.
    chosen_reg=None; reg_scores={}
    for reg in cfg["operator_regularization_grid"]:
        _, diag=fit_operator_bank(basis, basis_stats["prior_mean"], basis_stats["prior_covariance"],
            op_eps, sigma2=sigma2, seed=cfg["optimizer_seeds"][0],
            epochs=cfg.get("operator_regularization_selection_epochs", 40),
            lr=cfg["learning_rate"], regularization=float(reg), inner_episodes=inner_op)
        reg_scores[str(reg)]=diag["inner_nll"]
    chosen_reg=min(cfg["operator_regularization_grid"],key=lambda x:reg_scores[str(x)])
    code_order=np.random.default_rng(cfg["seed"]+900).permutation(4).tolist()
    opaque_map={str(physical):int(code_order[physical]) for physical in range(4)}
    # Remap physical training tags to randomized opaque IDs before the operator trainer sees them.
    op_eps_opaque=[ObservedEpisode(e.support_x,e.support_y,e.query_x,e.query_y,
         tuple(opaque_map[str(c)] for c in e.opaque_program)) for e in op_eps]
    inner_op_opaque=[ObservedEpisode(e.support_x,e.support_y,e.query_x,e.query_y,
         tuple(opaque_map[str(c)] for c in e.opaque_program)) for e in inner_op]
    banks={}; context_free_banks={}; prior_stats={}; operator_curves={}
    for opt_seed in cfg["optimizer_seeds"]:
        bank,diag=fit_operator_bank(basis,basis_stats["prior_mean"],basis_stats["prior_covariance"],
          op_eps_opaque,sigma2=sigma2,seed=opt_seed,epochs=cfg["operator_epochs"],
          lr=cfg["learning_rate"],regularization=float(chosen_reg),inner_episodes=inner_op_opaque)
        path=out/f"operator_seed{opt_seed}.pt"
        torch.save({"state_dict":bank.state_dict(),"opaque_code_map":opaque_map,
          "basis_sha256":sha256(out/"basis_pretrained.pt"),"regularization":chosen_reg},path)
        banks[opt_seed]=(bank,diag); prior_stats[opt_seed]=(basis_stats["prior_mean"],basis_stats["prior_covariance"],sigma2)
        operator_curves[str(opt_seed)]=diag
        free_bank, free_diag=fit_operator_bank(basis,basis_stats["prior_mean"],basis_stats["prior_covariance"],
          op_eps_opaque,sigma2=sigma2,seed=opt_seed+10000,epochs=cfg["operator_epochs"],
          lr=cfg["learning_rate"],regularization=float(chosen_reg),inner_episodes=inner_op_opaque,
          use_old_support=False)
        context_free_banks[opt_seed]=free_bank
        torch.save({"state_dict":free_bank.state_dict(),"opaque_code_map":opaque_map,
          "basis_sha256":sha256(out/"basis_pretrained.pt"),"regularization":chosen_reg,
          "context_ablated":True},out/f"context_ablated_operator_seed{opt_seed}.pt")
        operator_curves[f"context_ablated_{opt_seed}"]=free_diag
        print(f"Operator banks complete for optimizer seed {opt_seed}", flush=True)
    (out/"operator_training.json").write_text(json.dumps({"regularization_candidates_inner_nll":reg_scores,
        "selected_regularization":chosen_reg,"opaque_code_map":opaque_map,"curves":operator_curves},indent=2),encoding="utf-8")

    # Freeze a common length penalty using inner train backgrounds and permitted compositions only.
    penalty_rows=[]
    inner_eval_params=inner_params
    allowed_eval=[()] + [(i,) for i in range(4)] + [tuple(p) for p in cfg["composition_training_pairs"]]
    for penalty in cfg["train_inner_penalty_grid"]:
        correct=[]
        for bi,before in enumerate(inner_eval_params):
            for pi,program in enumerate(allowed_eval):
                old,adapt,_=stage_pair(before,program,seed=980000+bi*1009+pi,
                    n_support=cfg["old_support"],n_query=cfg["adaptation_trajectory"],dt=cfg["dt"],sigma=cfg["sigma"])
                classes,_=functional_classes(before,dt=cfg["dt"])
                for seed,(bank,_) in banks.items():
                    allowed_opaque=[tuple(opaque_map[str(c)] for c in p) for p in allowed_eval]
                    _,ranked=_scores_for_case(basis,bank,*prior_stats[seed][:2],sigma2,old,adapt,
                        allowed_opaque,min(20,cfg["adaptation_trajectory"]),float(penalty))
                    inv={v:int(k) for k,v in opaque_map.items()}
                    predicted=tuple(inv[c] for c in ranked[0][0])
                    correct.append(classes[PROGRAMS.index(predicted)] == classes[PROGRAMS.index(program)])
        penalty_rows.append({"penalty":float(penalty),"class_accuracy":float(np.mean(correct))})
    length_penalty=max(penalty_rows,key=lambda r:(r["class_accuracy"],-r["penalty"]))["penalty"]
    print(f"Train-inner length penalty frozen at {length_penalty}", flush=True)
    (out/"length_penalty_selection.json").write_text(json.dumps({"train_inner_only":True,
      "candidate_results":penalty_rows,"selected":length_penalty,"selection_programs":"old, singles, permitted ordered pairs"},indent=2),encoding="utf-8")

    dev_rows=evaluate_cases(basis,banks,context_free_banks,prior_stats,{"IDLIKE_DEVELOPMENT":id_params,
      "BACKGROUND_OOD_DEVELOPMENT":ood_params},{"IDLIKE_DEVELOPMENT":id_seeds,
      "BACKGROUND_OOD_DEVELOPMENT":ood_seeds},cfg,opaque_map,length_penalty,out,
      train_b_mean=float(np.mean([p.b for p in train_params])))
    import csv
    with (out/"per_background.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(dev_rows[0]));writer.writeheader();writer.writerows(dev_rows)
    by_cell={}
    for row in dev_rows:
        key=f"{row['regime']}|N={row['N']}"
        by_cell.setdefault(key,[]).append(row)
    metrics={key:{"n_background_program_seed_rows":len(vals),
        "functional_class_accuracy":float(np.mean([r["functional_class_correct"] for r in vals])),
        "mean_query_delta_v_mse":float(np.mean([r["query_delta_v_mse"] for r in vals])),
        "mean_true_vs_best_wrong_class_margin":float(np.mean([r["true_vs_best_wrong_class_margin"] for r in vals])),
        "mean_old_posterior_trace":float(np.mean([r["old_posterior_trace"] for r in vals])),
        "mean_selected_posterior_trace":float(np.mean([r["selected_posterior_trace"] for r in vals]))} for key,vals in by_cell.items()}
    (out/"metrics.json").write_text(json.dumps({"status":"DEVELOPMENT_DIAGNOSTIC",
        "length_penalty":length_penalty,"metrics":metrics,"limitations":[
        "This first executable v2 run is CPU development only; no original test splits used.",
        "Assessment query labels are used only after selection for query-MSE reporting.",
        "Legacy M2/M3/M4 paired reevaluation and privileged system-ID references are logged as pending if not emitted."]},indent=2),encoding="utf-8")
    runtime=time.time()-start
    provenance={**pretraining_provenance,"runtime_seconds":runtime,
        "source_hashes":{p.relative_to(ROOT).as_posix():sha256(p) for p in [ROOT/"q1_v2/basis.py",ROOT/"q1_v2/bayes.py",ROOT/"q1_v2/operators.py",ROOT/"q1_v2/inference.py",ROOT/"q1_v2/training.py",ROOT/"scripts/q1_v2_pipeline.py",config_path]},
        "checkpoint_hashes":{p.name:sha256(p) for p in out.glob("*.pt")},
        "optimizer_seeds":cfg["optimizer_seeds"],"legacy_checkpoint_availability":{
          "m2":(ROOT/"results/q1/cpu_confirmatory_impl/m2/context_model.pt").exists(),
          "m3":(ROOT/"results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt").exists(),
          "m4":(ROOT/"results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt").exists()},
        "results_rows":len(dev_rows)}
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(f"Completed Q1 v2 development run in {runtime:.1f}s: {out}")


if __name__ == "__main__":
    main()
