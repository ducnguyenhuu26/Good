"""Structural-identifiability and temporal-learning study for RWT Stage 2A.1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.rwt.run_stage2a import collect
from rwt.dynamics import default_regimes
from rwt.identifiability import composition_closure, fit_three_code_assignment
from rwt.stage2a import (
    ALL_THEORY_CODES, TRAIN_THEORY_CODES, CompositionalDynamics,
    IndependentExpertsDynamics, LinearStructuredDynamics, MonolithicDynamics,
    TheoryBelief, TransitionArrays, load_checkpoint, save_checkpoint,
    train_hmm_model, train_latent_model, train_privileged_factor_diagnostic,
)


METHODS = (
    "monolithic", "independent_experts", "composition_iid", "composition_hmm",
    "oracle_assignment", "linear_structured_iid", "linear_structured_hmm",
    "weak_calibration_hmm",
)


def arrays_for(rows: list[dict[str, object]]) -> TransitionArrays:
    return TransitionArrays(
        np.asarray([r["state"] for r in rows], np.float32),
        np.asarray([r["action"] for r in rows], np.float32),
        np.asarray([r["next_state"] for r in rows], np.float32),
    )


def grouped_sequences(rows: list[dict[str, object]]) -> list[tuple[int, TransitionArrays, list[dict[str, object]]]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["seed"])].append(row)
    return [(seed, arrays_for(sorted(group, key=lambda row: int(row["step"]))),
             sorted(group, key=lambda row: int(row["step"]))) for seed, group in sorted(groups.items())]


def concat_arrays(sequences: list[TransitionArrays]) -> TransitionArrays:
    return TransitionArrays(*(np.concatenate([getattr(x, field) for x in sequences], axis=0)
                              for field in ("state", "action", "next_state")))


def tensor_data(data: TransitionArrays, device: torch.device):
    return tuple(torch.as_tensor(x, dtype=torch.float32, device=device)
                 for x in (data.state, data.action, data.next_state))


def candidate_predictions(model, kind: str, state: torch.Tensor, action: torch.Tensor):
    if kind.startswith("composition") or kind == "oracle_assignment":
        means = state[:, None, :] + model.predict_codes(state, action)
        scales = model.log_scale.clamp(-6, 2).exp().view(1, 1, -1).expand_as(means)
    elif kind == "linear_structured_iid" or kind in {"linear_structured_hmm", "weak_calibration_hmm"}:
        means = state[:, None, :] + model.predict_codes(state, action)
        scales = model.log_scale.clamp(-6, 2).exp().view(1, 1, -1).expand_as(means)
    elif kind == "independent_experts":
        means = model.predict_codes(state, action)
        scales = model.log_scale.clamp(-6, 2).exp()[None, :, :].expand_as(means)
    else:
        mean, scale = model.predict(state, action)
        means, scales = mean[:, None, :], scale[:, None, :]
    return means, scales


def log_prob(target: torch.Tensor, means: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    z = (target[:, None, :] - means) / scales
    return -0.5 * (z.square() + 2.0 * scales.log() + np.log(2.0 * np.pi)).sum(-1)


def evaluate_stream(model, method: str, episode_rows: list[dict[str, object]], hazard: float = 0.04):
    """Causal prediction/filter followed by vectorized fixed-belief rollouts."""
    model.eval()
    device = next(model.parameters()).device
    states = torch.as_tensor(np.asarray([r["state"] for r in episode_rows]), dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.asarray([r["action"] for r in episode_rows]), dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.asarray([r["next_state"] for r in episode_rows]), dtype=torch.float32, device=device)
    with torch.no_grad():
        means, scales = candidate_predictions(model, method, states, actions)
        likelihoods = log_prob(next_states, means, scales).cpu().numpy().astype(np.float64)
        if method == "monolithic":
            codes = ((0, 0),)
        elif method == "independent_experts":
            codes = TRAIN_THEORY_CODES
        else:
            codes = ALL_THEORY_CODES
        belief = TheoryBelief(codes, hazard=hazard)
        prior_weights, posterior_weights, predicted, nlls = [], [], [], []
        mean_np = means.cpu().numpy()
        for i, ll in enumerate(likelihoods):
            weights = belief.predict_weights()
            prior_weights.append(weights)
            log_mix = np.log(weights.clip(1e-300)) + ll
            maximum = np.max(log_mix)
            nlls.append(float(-(maximum + np.log(np.exp(log_mix - maximum).sum()))))
            predicted.append(np.einsum("k,kd->d", weights, mean_np[i]))
            posterior_weights.append(belief.update(ll))
    predicted = np.asarray(predicted)
    truth = np.asarray([r["next_state"] for r in episode_rows])
    record_rows = []
    for i, row in enumerate(episode_rows):
        record = {"seed": int(row["seed"]), "step": int(row["step"]), "regime": str(row["regime"]),
                  "change_point": int(bool(row["change_point"])),
                  "mse": float(np.mean((predicted[i] - truth[i]) ** 2)), "nll": nlls[i],
                  "rollout_mse_h1": float(np.mean((predicted[i] - truth[i]) ** 2)),
                  "rollout_mse_h5": np.nan}
        for j in range(4):
            record[f"posterior_code_{j}"] = float(posterior_weights[i][j]) if j < len(codes) else 0.0
        record_rows.append(record)

    # Autoregressive rollout; vectorize all eligible anchors for each horizon.
    for horizon in (5,):
        anchors = [i for i in range(len(episode_rows) - horizon + 1)
                   if len({str(episode_rows[j]["regime"]) for j in range(i, i + horizon)}) == 1]
        if anchors:
            current = states[anchors].clone()
            weights = np.asarray(prior_weights)[anchors]
            with torch.no_grad():
                for offset in range(horizon):
                    ids = [i + offset for i in anchors]
                    state_t = current
                    action_t = actions[ids]
                    step_means, _ = candidate_predictions(model, method, state_t, action_t)
                    current = torch.as_tensor(np.einsum("nk,nkd->nd", weights, step_means.cpu().numpy()),
                                              dtype=torch.float32, device=device)
                targets = next_states[[i + horizon - 1 for i in anchors]]
                errors = ((current - targets) ** 2).mean(-1).cpu().numpy()
            for i, error in zip(anchors, errors):
                record_rows[i]["rollout_mse_h5"] = float(error)
    return record_rows


def episode_regime_metrics(trace_rows: list[dict[str, object]], method: str, budget: int, rep: int):
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in trace_rows:
        groups[(int(row["seed"]), str(row["regime"]))].append(row)
    summaries = []
    for (seed, regime), values in sorted(groups.items()):
        valid_rollouts = [float(x["rollout_mse_h5"]) for x in values if np.isfinite(x["rollout_mse_h5"])]
        summaries.append({"model": method, "budget_episodes": budget, "train_rep": rep,
                          "eval_seed": seed, "regime": regime,
                          "mse": float(np.mean([x["mse"] for x in values])),
                          "nll": float(np.mean([x["nll"] for x in values])),
                          "rollout_mse_h5": float(np.mean(valid_rollouts)) if valid_rollouts else np.nan,
                          "n_transitions": len(values), "n_rollouts_h5": len(valid_rollouts)})
    return summaries


def instantiate(method: str, hidden_dim: int):
    if method == "monolithic":
        return MonolithicDynamics(hidden_dim=max(12, int(hidden_dim * np.sqrt(3))))
    if method == "independent_experts":
        return IndependentExpertsDynamics(hidden_dim=hidden_dim)
    if method in {"composition_iid", "composition_hmm", "oracle_assignment"}:
        return CompositionalDynamics(hidden_dim=hidden_dim)
    return LinearStructuredDynamics()


def model_kind(method: str) -> str:
    if method == "monolithic":
        return "monolithic"
    if method == "independent_experts":
        return "experts"
    if method.startswith("linear") or method == "weak_calibration_hmm":
        return "linear"
    return "compositional"


def train_one(method: str, model, train_sequences, validation_sequences, train_flat,
              validation_flat, train_rows, validation_rows, *, epochs, seed, device, calibration=None,
              batch_size=256):
    kind = model_kind(method)
    train_regimes = np.asarray([r["regime"] for r in train_rows])
    validation_regimes = np.asarray([r["regime"] for r in validation_rows])
    if method in {"composition_hmm", "linear_structured_hmm", "weak_calibration_hmm"}:
        return train_hmm_model(model, train_sequences, validation_sequences, model_kind=kind,
                               epochs=epochs, seed=seed, device=device, hazard=0.04,
                               calibration=calibration if method == "weak_calibration_hmm" else None)
    if method == "oracle_assignment":
        return train_privileged_factor_diagnostic(model, train_flat, train_regimes,
                                                  validation_flat, validation_regimes,
                                                  epochs=epochs, seed=seed, device=device)
    iid_kind = "linear" if method == "linear_structured_iid" else kind
    return train_latent_model(model, train_flat, validation_flat, model_kind=iid_kind,
                              epochs=epochs, seed=seed, device=device, batch_size=batch_size)


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _bootstrap_ci(values: np.ndarray, *, seed: int = 99, reps: int = 5000):
    """Two-way paired bootstrap over training rep and fixed evaluation seed."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    rng = np.random.default_rng(seed)
    ri = rng.integers(0, values.shape[0], size=(reps, values.shape[0]))
    ei = rng.integers(0, values.shape[1], size=(reps, values.shape[1]))
    draws = values[ri[:, :, None], ei[:, None, :]].mean(axis=(1, 2))
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def aggregate_ci(rows, metric: str):
    groups: dict[tuple[str, int, str], dict[tuple[int, int], float]] = defaultdict(dict)
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(float(value)):
            groups[(str(row["model"]), int(row["budget_episodes"]), str(row["regime"]))][
                (int(row["train_rep"]), int(row["eval_seed"]))] = float(value)
    out = []
    for (model, budget, regime), mapping in sorted(groups.items()):
        reps = sorted({key[0] for key in mapping}); seeds = sorted({key[1] for key in mapping})
        matrix = np.asarray([[mapping[(r, s)] for s in seeds] for r in reps])
        lo, hi = _bootstrap_ci(matrix, seed=budget + sum(map(ord, model + regime)))
        out.append({"model": model, "budget_episodes": budget, "regime": regime, "metric": metric,
                    "mean": float(matrix.mean()), "std": float(matrix.reshape(-1).std(ddof=1)),
                    "ci95_low": lo, "ci95_high": hi, "train_reps": len(reps), "eval_seeds": len(seeds)})
    return out


def paired_vs_iid(rows, metric: str):
    index = {(str(r["model"]), int(r["budget_episodes"]), int(r["train_rep"]), int(r["eval_seed"]), str(r["regime"])): r
             for r in rows}
    output = []
    for method in sorted({str(r["model"]) for r in rows} - {"composition_iid", "stage2a_reference"}):
        budgets = sorted({int(r["budget_episodes"]) for r in rows if r["model"] == method})
        for budget in budgets:
            reps = sorted({int(r["train_rep"]) for r in rows if r["model"] == method and int(r["budget_episodes"]) == budget})
            seeds = sorted({int(r["eval_seed"]) for r in rows if r["model"] == method and int(r["budget_episodes"]) == budget})
            for regime in ("A", "B", "D", "C"):
                differences = np.full((len(reps), len(seeds)), np.nan)
                for ri, rep in enumerate(reps):
                    for si, eval_seed in enumerate(seeds):
                        key=(method,budget,rep,eval_seed,regime); base=("composition_iid",budget,rep,eval_seed,regime)
                        if key in index and base in index:
                            a=index[key].get(metric); b=index[base].get(metric)
                            if a is not None and b is not None and np.isfinite(float(a)) and np.isfinite(float(b)):
                                differences[ri,si]=float(a)-float(b)
                if np.isfinite(differences).any():
                    valid_reps = np.where(np.isfinite(differences).any(axis=1))[0]
                    valid_seeds = np.where(np.isfinite(differences).any(axis=0))[0]
                    matrix = differences[np.ix_(valid_reps, valid_seeds)]
                    matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix))
                    lo, hi = _bootstrap_ci(matrix, seed=budget + sum(map(ord, method + regime)))
                    output.append({"contrast": f"{method} - composition_iid", "budget_episodes": budget,
                                   "regime": regime, "metric": metric, "mean_paired_difference": float(matrix.mean()),
                                   "std_paired_difference": float(matrix.reshape(-1).std(ddof=1)),
                                   "ci95_low": lo, "ci95_high": hi, "n_pairs": int(np.isfinite(differences).sum())})
    return output


def paired_vs_stage2a_reference(rows, metric: str, full_budget: int):
    index={(str(r["model"]),int(r["budget_episodes"]),int(r["train_rep"]),int(r["eval_seed"]),str(r["regime"])):r
           for r in rows}
    methods=sorted({str(r["model"]) for r in rows}-{"stage2a_reference"})
    output=[]
    for method in methods:
        for regime in ("A","B","D","C"):
            diffs=[]
            for seed in sorted({int(r["eval_seed"]) for r in rows if r["model"]=="stage2a_reference"}):
                candidate=index.get((method,full_budget,0,seed,regime))
                reference=index.get(("stage2a_reference",full_budget,0,seed,regime))
                if candidate is None or reference is None: continue
                a,b=candidate.get(metric),reference.get(metric)
                if a is not None and b is not None and np.isfinite(float(a)) and np.isfinite(float(b)):
                    diffs.append(float(a)-float(b))
            if diffs:
                lo,hi=_bootstrap_ci(np.asarray(diffs),seed=full_budget+sum(map(ord,method+regime+metric)))
                output.append({"contrast":f"{method} - stage2a_reference(rep0)","budget_episodes":full_budget,
                               "regime":regime,"metric":metric,"mean_paired_difference":float(np.mean(diffs)),
                               "std_paired_difference":float(np.std(diffs,ddof=1)) if len(diffs)>1 else 0.0,
                               "ci95_low":lo,"ci95_high":hi,"n_pairs":len(diffs)})
    return output


def run_closure(train_rows, evaluation_rows, output_dir: Path, seed: int):
    rng = np.random.default_rng(seed)
    supports = []
    train_pick = rng.choice(len(train_rows), size=min(256, len(train_rows)), replace=False)
    supports.append(("train_support", [train_rows[i] for i in train_pick]))
    held = [row for row in evaluation_rows if row["regime"] == "C"]
    hold_pick = rng.choice(len(held), size=min(256, len(held)), replace=False)
    supports.append(("heldout_C_support", [held[i] for i in hold_pick]))
    true_params = {name: regime.parameters for name, regime in default_regimes().items()}
    point_rows=[]; summaries=[]
    for split, rows in supports:
        states=np.asarray([r["state"] for r in rows]); actions=np.asarray([r["action"] for r in rows])
        residual, relative=composition_closure(true_params, states, actions)
        norms=np.linalg.norm(residual,axis=-1)
        for i in range(len(rows)):
            point_rows.append({"support":split,"probe_index":i,"absolute_closure_norm":float(norms[i]),
                               "relative_closure_error":float(relative[i])})
        summaries.append({"support":split,"n_points":len(rows),"absolute_mean":float(norms.mean()),
                          "absolute_max":float(norms.max()),"relative_mean":float(relative.mean()),
                          "relative_max":float(relative.max())})
    write_csv(output_dir/"closure_probe_points.csv",point_rows)
    write_csv(output_dir/"closure_summary.csv",summaries)


def assignment_ambiguity_rows():
    observations={"A":np.asarray([0.0]),"B":np.asarray([1.0]),"D":np.asarray([3.0])}
    assignments={
        "anchor_A": {"A":(0,0),"B":(1,0),"D":(0,1)},
        "anchor_B": {"B":(0,0),"A":(1,0),"D":(0,1)},
    }
    rows=[]
    for name,mapping in assignments.items():
        reconstructed, heldout=fit_three_code_assignment(observations,mapping)
        rows.append({"assignment":name,"max_train_reconstruction_error":max(float(abs(reconstructed[k]-observations[k]).max()) for k in observations),
                     "predicted_code11":float(heldout[0])})
    return rows


def primitive_diagnostic(model, method, train_data, rep, budget):
    if method not in {"composition_iid","composition_hmm","oracle_assignment"} and not method.startswith("linear_structured") and method != "weak_calibration_hmm":
        return []
    n=min(128,len(train_data.state)); s=torch.as_tensor(train_data.state[:n],dtype=torch.float32,device=next(model.parameters()).device)
    a=torch.as_tensor(train_data.action[:n],dtype=torch.float32,device=next(model.parameters()).device)
    with torch.no_grad():
        if isinstance(model,LinearStructuredDynamics):
            p1=model.velocity_primitive(s[:,2:]); p2=model.action_primitive(a)
        else:
            p1=model.primitives[0](s,a); p2=model.primitives[1](s,a)
        v1=p1.flatten(1); v2=p2.flatten(1)
        cosine=torch.nn.functional.cosine_similarity(v1,v2,dim=-1)
    return [{"model":method,"train_rep":rep,"budget_episodes":budget,
             "primitive1_mean_norm":float(v1.norm(dim=-1).mean()),
             "primitive2_mean_norm":float(v2.norm(dim=-1).mean()),
             "mean_cosine_similarity":float(cosine.mean()),
             "mean_abs_cosine_similarity":float(cosine.abs().mean())}]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results/rwt/stage2a1")
    parser.add_argument("--budgets",default="1,4,12",help="number of training episodes per budget")
    parser.add_argument("--replications",type=int,default=3)
    parser.add_argument("--epochs",type=int,default=28)
    parser.add_argument("--hidden-dim",type=int,default=32)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--calibration-transitions",type=int,default=8)
    parser.add_argument("--methods",default=",".join(METHODS),help="comma-separated Stage 2A.1 ablations")
    args=parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    budgets=sorted({int(x) for x in args.budgets.split(",")})
    methods=tuple(x.strip() for x in args.methods.split(",") if x.strip())
    if not methods or set(methods)-set(METHODS):
        raise SystemExit(f"methods must be selected from {METHODS}")
    if budgets[0]<1 or args.replications<2:
        raise SystemExit("require positive data budgets and at least two training replications")
    val_seeds=[142,143,144]; eval_seeds=[242,243,244,245,246,247]
    validation,validation_rows=collect("train",val_seeds)
    val_sequences=[x[1] for x in grouped_sequences(validation_rows)]
    evaluation,evaluation_rows=collect("evaluation",eval_seeds)
    all_metrics=[]; all_traces=[]; all_curves=[]; all_occupancy=[]; all_primitives=[]; budget_manifest=[]
    fixed_eval_group=grouped_sequences(evaluation_rows)
    for rep in range(args.replications):
        first_seed=42+1000*rep
        full_train,full_rows=collect("train",list(range(first_seed,first_seed+max(budgets))))
        train_groups=grouped_sequences(full_rows)
        for budget in budgets:
            chosen=train_groups[:budget]
            train_sequences=[x[1] for x in chosen]
            train_rows=[row for _,_,episode in chosen for row in episode]
            train_flat=concat_arrays(train_sequences)
            train_regime_labels=np.asarray([row["regime"] for row in train_rows])
            assert "C" not in set(train_regime_labels)
            assert not any(row["regime"]=="C" for row in validation_rows)
            calibration_rows=[]
            for _,_,episode in chosen:
                calibration_rows.extend([row for row in episode if row["regime"]=="A" and int(row["step"])<args.calibration_transitions])
            calibration=arrays_for(calibration_rows) if calibration_rows else None
            budget_manifest.append({"train_rep":rep,"budget_episodes":budget,
                                    "train_seed_ids":";".join(str(x[0]) for x in chosen),
                                    "train_transitions":len(train_flat.state),"validation_transitions":len(validation.state),
                                    "evaluation_transitions":len(evaluation.state),"train_regimes":"A;B;D",
                                    "validation_regimes":"A;B;D","evaluation_regimes":"A;B;C;D"})
            for mi,method in enumerate(methods):
                init_seed=20_260+rep*10_000+budget*100+mi
                torch.manual_seed(init_seed)
                model=instantiate(method,args.hidden_dim)
                if isinstance(model,LinearStructuredDynamics):
                    model.fit_input_normalization(train_flat.state,train_flat.action)
                train_seed=60_260+rep*10_000+budget*100+mi
                history=train_one(method,model,train_sequences,val_sequences,train_flat,validation,
                                  train_rows,validation_rows,epochs=args.epochs,seed=train_seed,
                                  device=args.device,calibration=calibration,batch_size=args.batch_size)
                for h in history:
                    all_curves.append({"model":method,"train_rep":rep,"budget_episodes":budget,**h})
                if budget==max(budgets):
                    model_dir=args.output_dir/"checkpoints"; model_dir.mkdir(exist_ok=True)
                    save_checkpoint(model_dir/f"rep{rep}_{method}.pt",model,model_kind=model_kind(method),
                                    config={"hidden_dim":(int(args.hidden_dim*np.sqrt(3)) if method=="monolithic" else args.hidden_dim)}
                                    if model_kind(method)!="linear" else {"state_dim":4,"action_dim":2})
                traces=[]
                for eval_seed,_,episode in fixed_eval_group:
                    traces.extend(evaluate_stream(model,method,episode))
                for row in traces:
                    all_traces.append({"model":method,"budget_episodes":budget,"train_rep":rep,**row})
                all_metrics.extend(episode_regime_metrics(traces,method,budget,rep))
                all_primitives.extend(primitive_diagnostic(model,method,train_flat,rep,budget))
                for regime in ("A","B","C","D"):
                    subset=[r for r in traces if r["regime"]==regime]
                    if not subset: continue
                    occupancy=[np.mean([r[f"posterior_code_{j}"] for r in subset]) for j in range(4)]
                    entropy=-sum(p*np.log(max(p,1e-12)) for p in occupancy)/np.log(4)
                    all_occupancy.append({"model":method,"train_rep":rep,"budget_episodes":budget,
                                          "regime":regime,"code_00":occupancy[0],"code_10":occupancy[1],
                                          "code_01":occupancy[2],"code_11":occupancy[3],
                                          "normalized_posterior_entropy":float(entropy),
                                          "active_codes_gt_5pct":sum(p>.05 for p in occupancy)})
            print(f"finished train_rep={rep}, budget={budget} episodes ({len(train_flat.state)} transitions)",flush=True)
    # Historical accepted Stage 2A checkpoint, evaluated on exactly the same fixed test streams.
    reference_path=ROOT/"results/rwt/stage2a/compositional.pt"
    if reference_path.exists():
        reference,_=load_checkpoint(reference_path,map_location=args.device)
        reference_traces=[]
        for _,_,episode in fixed_eval_group:
            reference_traces.extend(evaluate_stream(reference,"composition_iid",episode))
        for row in episode_regime_metrics(reference_traces,"stage2a_reference",max(budgets),0):
            all_metrics.append(row)
        for row in reference_traces:
            all_traces.append({"model":"stage2a_reference","budget_episodes":max(budgets),"train_rep":0,**row})
    run_closure(
        [row for row in collect("train",list(range(42,42+max(budgets))))[1]],
        evaluation_rows,args.output_dir,seed=7301,
    )
    write_csv(args.output_dir/"training_manifest.csv",budget_manifest)
    write_csv(args.output_dir/"per_episode_metrics.csv",all_metrics)
    write_csv(args.output_dir/"evaluation_trace.csv",all_traces)
    write_csv(args.output_dir/"training_curves.csv",all_curves)
    write_csv(args.output_dir/"posterior_occupancy.csv",all_occupancy)
    write_csv(args.output_dir/"primitive_diversity.csv",all_primitives)
    write_csv(args.output_dir/"assignment_ambiguity.csv",assignment_ambiguity_rows())
    summaries=[]
    for metric in ("mse","nll","rollout_mse_h5"):
        summaries.extend(aggregate_ci(all_metrics,metric))
    write_csv(args.output_dir/"metrics_summary_ci.csv",summaries)
    paired=[]
    for metric in ("mse","nll","rollout_mse_h5"):
        paired.extend(paired_vs_iid(all_metrics,metric))
        paired.extend(paired_vs_stage2a_reference(all_metrics,metric,max(budgets)))
    write_csv(args.output_dir/"paired_comparisons.csv",paired)
    with (args.output_dir/"config.json").open("w",encoding="utf-8") as handle:
        json.dump({"budgets_episodes":budgets,"replications":args.replications,
                   "train_seed_bases":[42+1000*r for r in range(args.replications)],
                   "validation_seeds":val_seeds,"fixed_evaluation_seeds":eval_seeds,
                   "epochs":args.epochs,"hidden_dim":args.hidden_dim,"device":args.device,
                   "hmm_hazard":.04,"calibration_transitions_per_episode":args.calibration_transitions,
                   "methods":list(methods),"c_excluded_from_training_validation_and_selection":True},handle,indent=2)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    colors=plt.cm.viridis(np.linspace(.05,.95,len(methods)))
    for color,method in zip(colors,methods):
        vals=[r for r in all_curves if r["model"]==method and r["budget_episodes"]==max(budgets)]
        if vals:
            epochs=sorted({int(r["epoch"]) for r in vals})
            means=[np.mean([r["validation_nll"] for r in vals if int(r["epoch"])==e]) for e in epochs]
            axes[0].plot(epochs,means,label=method,color=color)
    axes[0].set(xlabel="Epoch",ylabel="A/B/D validation NLL",title="Full-budget learning curves")
    axes[0].legend(fontsize=6)
    for color,method in zip(colors,methods):
        values=[r for r in summaries if r["model"]==method and r["regime"]=="C" and r["metric"]=="mse"]
        values.sort(key=lambda r:r["budget_episodes"])
        if values:
            x=[r["budget_episodes"] for r in values]; y=[r["mean"] for r in values]
            axes[1].plot(x,y,marker="o",label=method,color=color)
            axes[1].fill_between(x,[r["ci95_low"] for r in values],[r["ci95_high"] for r in values],alpha=.12,color=color)
    axes[1].set(xlabel="Training episodes",ylabel="Held-out C one-step MSE",title="Data scaling (C never selects models)")
    axes[1].legend(fontsize=6); fig.tight_layout(); fig.savefig(args.output_dir/"stage2a1_data_scaling.png",dpi=160); plt.close(fig)
    print(f"Wrote Stage 2A.1 artifacts to {args.output_dir}")
    print(f"models={len(methods)} budgets={budgets} replications={args.replications} fixed_eval_seeds={eval_seeds}")
    for row in summaries:
        if row["regime"]=="C" and row["metric"]=="mse": print(json.dumps(row,sort_keys=True))


if __name__=="__main__":
    main()
