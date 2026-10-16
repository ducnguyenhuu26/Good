"""Stage 2C: online theory revision and recurrence memory benchmark.

Training and calibration use A/B/D only. Test streams are consumed one
transition at a time; labels/switch times are evaluator sidecars only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

from experiments.rwt.run_stage2a import collect
from experiments.rwt.run_stage2b import evaluator_alignment
from rwt.agents import constrain_action
from rwt.dynamics import default_regimes
from rwt.environment import RecurrentDynamicsEnv
from rwt.schedule import RegimeSchedule, Segment
from rwt.stage2a import IndependentExpertsDynamics, MonolithicDynamics, gaussian_log_prob, load_checkpoint
from rwt.stage2b import (OnlineStructuralAgent, learned_compositional_prediction_set,
                         learned_expert_prediction_set, learned_monolithic_prediction_set,
                         oracle_prediction_set)
from rwt.stage2c import OnlineTheoryRevision, RLSResidual, StreamingTheoryAdapters

SEGMENTS = (Segment("A", 40), Segment("B", 40), Segment("D", 40),
            Segment("C", 48), Segment("A", 40), Segment("C", 48))
MAX_ACTION = .75
SEEDS = tuple(range(3300, 3320))


def write_csv(path: Path, rows: list[dict]):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            fields=list(rows[0])
            for row in rows:
                fields.extend(k for k in row if k not in fields)
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def load_models():
    root = ROOT / "results/rwt/stage2b/learned/checkpoints"
    expert, _ = load_checkpoint(root / "independent_experts.pt")
    mono, _ = load_checkpoint(root / "monolithic.pt")
    return expert, mono


def collect_stream(seed: int):
    """Generate common, policy-independent transitions and evaluator labels."""
    schedule = RegimeSchedule(SEGMENTS)
    env = RecurrentDynamicsEnv(schedule, noise_std=.025, target=(2., 2.), max_action_norm=MAX_ACTION)
    state, _ = env.reset(seed=seed + 50_000)
    rows = []
    for t in range(schedule.horizon):
        action = constrain_action(.30 * (np.asarray((2., 2.)) - state[:2]) - .72 * state[2:], MAX_ACTION)
        next_state, reward, terminated, truncated, _ = env.step(action)
        rows.append({"seed": seed, "step": t, "regime_eval_only": schedule.regime_name_at(t),
                     "change_point_eval_only": int(t in schedule.change_steps()),
                     "state": state.copy(), "action": action.copy(), "next_state": next_state.copy(),
                     "reward": float(reward), "terminated": int(terminated), "truncated": int(truncated)})
        state = next_state
        if terminated or truncated: break
    env.close()
    return rows


def prediction_factories(expert, mono):
    return {
        "composition": lambda s, a: learned_compositional_prediction_set(expert, s, a),
        "experts": lambda s, a: learned_expert_prediction_set(expert, s, a),
        "monolithic": lambda s, a: learned_monolithic_prediction_set(mono, s, a),
    }


def calibrate_threshold(factory, known_count: int, completion_indices: tuple[int, ...]):
    """Calibrate standardized innovation cutoff using A/B/D validation only."""
    data, sidecar = collect("train", [142, 143, 144])
    if any(r["regime"] == "C" for r in sidecar):
        raise AssertionError("C leaked into threshold calibration")
    scores = []
    for state, action, target in zip(data.state, data.action, data.next_state):
        pred = factory(state, action)
        z = np.sqrt(np.mean(((target[2:][None] - pred.means) / pred.scales) ** 2, axis=-1))
        scores.append(float(np.min(z)))
    return float(np.quantile(scores, .995)), scores


def audit_stage2b(expert, train_data, seeds):
    """Re-run Stage 2B policies to separate label alignment from predictive fit."""
    from rwt.stage2b import OnlineStructuralAgent
    regimes = default_regimes()
    models = tuple(regimes[n].parameters for n in ("A", "B", "D"))
    align = evaluator_alignment(expert, train_data, "composition")
    factory = lambda s, a: learned_compositional_prediction_set(expert, s, a)
    audit = []; pairs = []; recurrence = []
    names = ("expert_0", "expert_1", "expert_2", "completion_0", "completion_1", "completion_2")
    for seed in seeds:
        for policy in ("passive", "active"):
            schedule = RegimeSchedule(SEGMENTS)
            env = RecurrentDynamicsEnv(schedule, noise_std=.025, target=(2.,2.), max_action_norm=MAX_ACTION)
            state,_=env.reset(seed=seed+50_000)
            agent=OnlineStructuralAgent(factory,names=names,known_count=3,completion_indices=(3,4,5),
                policy=policy,seed=seed+70_000,horizon=schedule.horizon,probing_budget=24,
                max_action_norm=MAX_ACTION,hazard=.05,use_memory=True)
            for step in range(schedule.horizon):
                regime=schedule.regime_name_at(step); target=align.get(regime,-1)
                action=agent.act(state,step); next_state,reward,terminated,truncated,_=env.step(action)
                obs=agent.observe(state,action,next_state)
                per=np.mean((np.concatenate((np.repeat(state[:2][None,:],6,axis=0)+obs.predictions.means,
                                              obs.predictions.means),axis=1)-next_state[None,:])**2,axis=1)
                map_idx=int(np.argmax(obs.posterior)); best_idx=int(np.argmin(per))
                z=(next_state[2:][None,:]-obs.predictions.means)/obs.predictions.scales
                true_z=z[target] if target>=0 else np.array([np.nan,np.nan])
                behavior=float(map_idx==best_idx)
                audit.append({"seed":seed,"policy":policy,"step":step,"regime_eval_only":regime,
                    "label_based_accuracy":float(map_idx==target) if target>=0 else np.nan,
                    "behavioral_identification_accuracy":behavior,
                    "map_candidate_mse":float(per[map_idx]),"best_candidate_mse":float(per[best_idx]),
                    "aligned_candidate_mse":float(per[target]) if target>=0 else np.nan,
                    "target_standardized_residual_rms":float(np.sqrt(np.mean(true_z**2))) if target>=0 else np.nan,
                    "target_coverage_68":float(np.mean(np.abs(true_z)<=1)) if target>=0 else np.nan,
                    "target_coverage_95":float(np.mean(np.abs(true_z)<=1.96)) if target>=0 else np.nan,
                    "posterior_target":float(obs.posterior[target]) if target>=0 else np.nan,
                    "prior_target":float(obs.prior[target]) if target>=0 else np.nan,
                    "target_in_memory":int(target in agent.filter.remembered) if target>=0 else 0,
                    "nll_preupdate":obs.nll,"mse_preupdate":obs.mse})
                # Per-pair predictive separation; useful to see if label errors
                # are caused by candidates too close to resolve behaviorally.
                diffs=[]
                for i in range(len(obs.predictions.names)):
                    for j in range(i+1,len(obs.predictions.names)):
                        d=obs.predictions.means[i]-obs.predictions.means[j]
                        scale=np.sqrt(obs.predictions.scales[i]**2+obs.predictions.scales[j]**2)
                        diffs.append((i,j,float(np.sqrt(np.mean(d*d))),float(np.sqrt(np.mean((d/scale)**2)))))
                pairs.append({"seed":seed,"policy":policy,"step":step,"regime_eval_only":regime,
                    "mean_pairwise_velocity_distance":float(np.mean([d[2] for d in diffs])),
                    "min_pairwise_velocity_distance":float(np.min([d[2] for d in diffs])),
                    "mean_pairwise_noise_normalized_distance":float(np.mean([d[3] for d in diffs]))})
                if regime=="C" and step in (120,121,122,124,128,136,152,167,208,209,210,212,216,224,240,255):
                    recurrence.append({"seed":seed,"policy":policy,"step":step,"appearance":"first" if step<168 else "return",
                        "offset":step-(120 if step<168 else 208),"prior_target":float(obs.prior[target]),
                        "posterior_target":float(obs.posterior[target]),"cache_contains_target":int(target in agent.filter.remembered),
                        "novel_prior_mass":float(obs.prior[list(agent.filter.completion_indices)].sum()),
                        "mse_preupdate":obs.mse,"nll_preupdate":obs.nll})
                state=next_state
                if terminated or truncated: break
            env.close()
    return audit,pairs,recurrence


def make_revision(method, factories, thresholds):
    if method == "compositional_frozen":
        factory, anchors, th, memory = factories["composition"], (3, 4, 5), 1e9, False
    elif method.startswith("compositional_revision"):
        factory, anchors, th = factories["composition"], (3, 4, 5), thresholds["composition"]
        memory = method.endswith("memory")
    elif method == "independent_experts_memory":
        factory, anchors, th, memory = factories["experts"], (0, 1, 2), thresholds["experts"], True
    else:
        raise ValueError(method)
    pred0 = factory(np.zeros(4), np.zeros(2))
    return OnlineTheoryRevision(factory, pred0.names, completion_indices=anchors,
                                mismatch_threshold=th, memory=memory)


class SlidingWindowResidual:
    """Frozen monolithic backbone plus a refit, bounded-window 10-weight adapter."""
    def __init__(self, model, window=24, ridge=8.0):
        self.model = model; self.window = int(window); self.ridge = float(ridge)
        self.data = []; self.adapter = RLSResidual(ridge=ridge)
        self.seconds = 0.0

    def base(self, state, action):
        s = torch.as_tensor(np.asarray(state, np.float32)[None]); a = torch.as_tensor(np.asarray(action, np.float32)[None])
        with torch.no_grad(): mean, scale = self.model.predict(s, a)
        return mean[0].numpy().astype(float), scale[0].numpy().astype(float)

    def predict(self, state, action):
        mean, scale = self.base(state, action)
        correction = self.adapter.predict(state, action)
        velocity = mean[2:] + correction
        return velocity, scale[2:]

    def observe(self, state, action, next_state):
        mean, _ = self.base(state, action)
        self.data.append((state.copy(), action.copy(), next_state[2:] - mean[2:]))
        self.data = self.data[-self.window:]
        tic = time.perf_counter()
        if self.data:
            from rwt.stage2c import residual_features
            x = np.stack([residual_features(s, a) for s, a, _ in self.data])
            y = np.stack([r for _, _, r in self.data])
            self.adapter.weights = np.linalg.solve(x.T @ x + self.ridge * np.eye(x.shape[1]), x.T @ y)
            self.adapter.updates = len(self.data)
        self.seconds += time.perf_counter() - tic


class FullFineTune:
    """Monolithic online SGD, one gradient step per observed transition."""
    def __init__(self, model, lr=3e-4):
        self.model = deepcopy(model); self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.seconds = 0.0; self.updates = 0

    def predict(self, state, action):
        s = torch.as_tensor(np.asarray(state, np.float32)[None]); a = torch.as_tensor(np.asarray(action, np.float32)[None])
        with torch.no_grad(): mean, scale = self.model.predict(s, a)
        return mean[0].numpy().astype(float), scale[0].numpy().astype(float)

    def observe(self, state, action, next_state):
        tic = time.perf_counter()
        s = torch.as_tensor(np.asarray(state, np.float32)[None]); a = torch.as_tensor(np.asarray(action, np.float32)[None])
        sn = torch.as_tensor(np.asarray(next_state, np.float32)[None])
        mean, scale = self.model.predict(s, a)
        loss = -gaussian_log_prob(sn, mean, scale).mean()
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.)
        self.optimizer.step(); self.updates += 1
        self.seconds += time.perf_counter() - tic


def _map_alignment_for_candidate(method, manager, step_result, alignments):
    if method == "independent_experts_memory":
        mapping = alignments["experts"]
    else:
        mapping = alignments["composition"]
    if step_result.map_index < len(mapping):
        return step_result.map_index
    # A revision inherits its nearest predictive anchor; the evaluator uses
    # this ancestry solely for label accuracy, never for retrieval/update.
    candidate = manager.last_candidates[step_result.map_index]
    return int(candidate[0])


def evaluate_revision(method, streams, factories, thresholds, alignments, mono):
    per_seed, traces = [], []
    for rows in streams:
        seed = int(rows[0]["seed"])
        manager = make_revision(method, factories, thresholds) if method in {
            "compositional_frozen", "compositional_revision", "compositional_revision_memory",
            "independent_experts_memory"} else None
        full = FullFineTune(deepcopy(mono)) if method == "monolithic_full_finetune" else None
        window = SlidingWindowResidual(deepcopy(mono)) if method == "sliding_window" else None
        output = []
        for row in rows:
            step_tic = time.perf_counter()
            s, a, sn = row["state"], row["action"], row["next_state"]
            target_regime = row["regime_eval_only"]
            if manager is not None:
                base, means, scales, candidates, prior, pred_v = manager.predict(s, a)
                pred_next = np.r_[s[:2] + pred_v, pred_v]
                # Score before observing this transition.
                result = manager.observe(s, a, sn)
                target_index = alignments["experts" if method == "independent_experts_memory" else "composition"].get(target_regime, -1)
                mapped = _map_alignment_for_candidate(method, manager, result, alignments)
                label_acc = float(mapped == target_index) if target_index >= 0 else np.nan
                behavioral = result.behavioral_accuracy
                mse, nll = float(np.mean((pred_next - sn) ** 2)), result.nll
                alarm, created, cache_hit = result.mismatch_alarm, result.created, result.cache_hit
                update_n = result.active_updates
                memory_size = result.memory_size
                ep, al = result.epistemic_variance, result.aleatoric_variance
                score = result.mismatch_score
                parameter_budget = 10 if method != "compositional_frozen" else 0
                update_applied = int(result.active_updates > 0)
            elif full is not None:
                mean, scale = full.predict(s, a)
                pred_next = np.r_[s[:2] + mean[2:], mean[2:]]
                err = (sn[2:] - mean[2:]) / scale[2:]
                nll = float((.5 * (err**2 + 2*np.log(scale[2:]) + np.log(2*np.pi))).sum())
                mse = float(np.mean((pred_next - sn) ** 2)); score = float(np.sqrt(np.mean(err**2)))
                alarm, created, cache_hit, update_n, memory_size = False, False, False, full.updates, 0
                ep, al, behavioral, label_acc = 0.0, float(np.mean(scale[2:]**2)), np.nan, np.nan
                parameter_budget = sum(p.numel() for p in full.model.parameters())
                update_applied = 1
                full.observe(s, a, sn)
            else:
                mean_v, scale_v = window.predict(s, a)
                pred_next = np.r_[s[:2] + mean_v, mean_v]
                err = (sn[2:] - mean_v) / scale_v
                nll = float((.5 * (err**2 + 2*np.log(scale_v) + np.log(2*np.pi))).sum())
                mse = float(np.mean((pred_next - sn) ** 2)); score = float(np.sqrt(np.mean(err**2)))
                alarm, created, cache_hit, update_n, memory_size = False, False, False, len(window.data), 0
                ep, al, behavioral, label_acc = 0.0, float(np.mean(scale_v**2)), np.nan, np.nan
                parameter_budget = 10
                update_applied = 1
                window.observe(s, a, sn)
            output.append({"method": method, "seed": seed, "step": row["step"],
                           "regime_eval_only": target_regime, "segment_offset": segment_offset(int(row["step"])),
                           "change_point_eval_only": row["change_point_eval_only"],
                           "preupdate_mse": mse, "preupdate_nll": nll,
                           "label_based_accuracy": label_acc, "behavioral_identification_accuracy": behavioral,
                           "mismatch_score": score, "mismatch_alarm": int(alarm),
                           "theory_created": int(created), "cache_hit": int(cache_hit),
                           "adapter_updates": update_n, "memory_size": memory_size,
                           "epistemic_variance": ep, "aleatoric_variance": al,
                           "reward": row["reward"], "compute_seconds": time.perf_counter()-step_tic,
                           "online_parameter_budget": parameter_budget,
                           "online_update_applied": update_applied})
        # Annotate cache success causally from evaluator labels: whether a
        # memory theory first created during the first C segment is selected
        # within 4 transitions on the second C occurrence.
        c_first = [x for x in output if x["regime_eval_only"] == "C" and x["step"] < 208]
        c_return = [x for x in output if x["regime_eval_only"] == "C" and x["step"] >= 208]
        created_first = any(x["theory_created"] for x in c_first)
        first_hit = next((int(x["step"] - c_return[0]["step"] + 1) for x in c_return[:4] if x["cache_hit"]), np.nan) if c_return else np.nan
        for x in output:
            x["first_C_revision_created_eval"] = int(created_first)
            x["recurrence_cache_retrieval_delay_eval"] = first_hit
        traces.extend(output)
        per_seed.append(aggregate_episode(method, seed, output, rows))
    return per_seed, traces


def evaluate_streaming_adapters(method,streams,factories,alignments):
    memory=method.endswith("memory")
    names=("expert_0","expert_1","expert_2","completion_0","completion_1","completion_2")
    per_seed=[]; traces=[]
    for rows in streams:
        seed=int(rows[0]["seed"])
        agent=StreamingTheoryAdapters(factories["composition"],names,memory=memory)
        output=[]; first_c_theories=set()
        for row in rows:
            tic=time.perf_counter(); s,a,sn=row["state"],row["action"],row["next_state"]
            base,means,scales,prior,pred_v,_,_=agent.predict(s,a)
            result=agent.observe(sn)
            regime=row["regime_eval_only"]; target=alignments["composition"].get(regime,-1)
            label=float(result.map_index==target) if target>=0 else np.nan
            cache_hit=int(memory and int(row["step"])>=208 and result.selected_index in first_c_theories)
            false_retrieval=int(memory and regime!="C" and result.selected_index in first_c_theories)
            if 120<=int(row["step"])<168:
                first_c_theories.add(result.selected_index)
            output.append({"method":method,"seed":seed,"step":row["step"],"regime_eval_only":regime,
                "segment_offset":segment_offset(int(row["step"])),"change_point_eval_only":row["change_point_eval_only"],
                "preupdate_mse":result.mse,"preupdate_nll":result.nll,"label_based_accuracy":label,
                "behavioral_identification_accuracy":result.behavioral_accuracy,"mismatch_score":result.mismatch_score,
                "mismatch_alarm":0,"theory_created":0,"cache_hit":cache_hit,"adapter_updates":1,
                "memory_size":len(agent.adapters),"epistemic_variance":result.epistemic_variance,
                "aleatoric_variance":result.aleatoric_variance,"reward":row["reward"],
                "compute_seconds":time.perf_counter()-tic,"online_parameter_budget":10,
                "online_update_applied":1,"selected_candidate":result.selected_index,
                "false_retrieval":false_retrieval})
        c_first=[x for x in output if x["regime_eval_only"]=="C" and x["step"]<208]
        c_return=[x for x in output if x["regime_eval_only"]=="C" and x["step"]>=208]
        first_hit=next((int(x["step"]-c_return[0]["step"]+1) for x in c_return[:4] if x["cache_hit"]),np.nan) if c_return else np.nan
        for x in output:
            x["first_C_revision_created_eval"]=0
            x["recurrence_cache_retrieval_delay_eval"]=first_hit
        traces.extend(output); per_seed.append(aggregate_episode(method,seed,output,rows))
    return per_seed,traces


def segment_offset(step):
    starts = (0, 40, 80, 120, 168, 208)
    idx = max(i for i, x in enumerate(starts) if step >= x)
    return step - starts[idx]


def aggregate_episode(method, seed, rows, stream):
    record = {"method": method, "seed": seed,
              "cumulative_return": float(sum(x["reward"] for x in stream)),
              "adaptation_samples_used": int(sum(x["online_update_applied"] for x in rows)),
              "revision_events": int(sum(x["theory_created"] for x in rows)),
              "cache_hit_transitions": int(sum(x["cache_hit"] for x in rows)),
              "memory_size_final": int(rows[-1]["memory_size"]),
              "compute_seconds": float(sum(x["compute_seconds"] for x in rows)),
              "online_parameter_budget": int(rows[0]["online_parameter_budget"])}
    for reg in ("A", "B", "D", "C"):
        subset = [r for r in rows if r["regime_eval_only"] == reg]
        record[f"mse_{reg}"] = float(np.mean([r["preupdate_mse"] for r in subset]))
        record[f"nll_{reg}"] = float(np.mean([r["preupdate_nll"] for r in subset]))
        acc = [r["label_based_accuracy"] for r in subset if np.isfinite(r["label_based_accuracy"])]
        beh = [r["behavioral_identification_accuracy"] for r in subset if np.isfinite(r["behavioral_identification_accuracy"])]
        record[f"label_accuracy_{reg}"] = float(np.mean(acc)) if acc else np.nan
        record[f"behavioral_accuracy_{reg}"] = float(np.mean(beh)) if beh else np.nan
    for app, lo, hi in (("first_C", 120, 168), ("recurrence_C", 208, 256), ("A_recurrence", 168, 208)):
        sub = [r for r in rows if lo <= r["step"] < hi]
        record[f"mse_{app}"] = float(np.mean([r["preupdate_mse"] for r in sub]))
        record[f"nll_{app}"] = float(np.mean([r["preupdate_nll"] for r in sub]))
        alarm_delay = next((r["step"] - lo + 1 for r in sub if r["mismatch_alarm"]), hi-lo)
        record[f"detection_delay_{app}"] = alarm_delay
    # Post hoc recovery time relative to this seed's late-first-C error level.
    # This is an online prediction trace metric; it never feeds the learner.
    first_c=[r for r in rows if 120<=r["step"]<168]
    rec_c=[r for r in rows if 208<=r["step"]<256]
    stable=float(np.mean([r["preupdate_mse"] for r in first_c[-8:]])) if first_c else np.nan
    recovered=48
    if np.isfinite(stable) and stable>0:
        for i in range(0,max(1,len(rec_c)-3)):
            if np.mean([r["preupdate_mse"] for r in rec_c[i:i+4]]) <= 1.25*stable:
                recovered=i+1; break
    record["model_recovery_delay_recurrence"] = recovered
    known = [r for r in rows if r["regime_eval_only"] in {"A", "B", "D"}]
    record["false_alarm_rate_known"] = float(np.mean([r["mismatch_alarm"] for r in known]))
    record["false_retrieval_rate_known"] = float(np.mean([r.get("false_retrieval",0) for r in known]))
    rec = [r for r in rows if r["step"] >= 208 and r["step"] < 256]
    record["recurrence_retrieval_delay"] = next((r["step"] - 208 + 1 for r in rec if r["cache_hit"]), 48)
    record["forgetting_A_mse_delta"] = record["mse_A_recurrence"] - record["mse_A"]
    return record


def _summary(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows: grouped[tuple(row[k] for k in keys)].append(row)
    out=[]
    for key, vals in grouped.items():
        r=dict(zip(keys,key)); r["n"]=len(vals)
        for metric in metrics:
            x=np.asarray([v[metric] for v in vals],float); x=x[np.isfinite(x)]
            r[metric+"_mean"] = float(x.mean()) if len(x) else np.nan
            r[metric+"_sd"] = float(x.std(ddof=1)) if len(x)>1 else 0.0 if len(x) else np.nan
            if len(x):
                rng=np.random.default_rng(717+len(metric)); draws=x[rng.integers(0,len(x),(5000,len(x)))].mean(1)
                r[metric+"_ci95_low"],r[metric+"_ci95_high"]=map(float,np.quantile(draws,[.025,.975]))
        out.append(r)
    return sorted(out,key=lambda r:tuple(str(r[k]) for k in keys))


def paired(rows, metrics):
    idx={(r["method"],int(r["seed"])):r for r in rows}; names=sorted({r["method"] for r in rows}); out=[]
    for i,m1 in enumerate(names):
        for m2 in names[i+1:]:
            ds={met:[] for met in metrics}
            for seed in SEEDS:
                a,b=idx.get((m1,seed)),idx.get((m2,seed))
                if a and b:
                    for met in metrics:
                        if np.isfinite(a[met]) and np.isfinite(b[met]): ds[met].append(a[met]-b[met])
            for met,x in ds.items():
                if not x: continue
                arr=np.asarray(x); rng=np.random.default_rng(912+sum(map(ord,m1+m2+met)))
                boots=arr[rng.integers(0,len(arr),(5000,len(arr)))].mean(1)
                out.append({"contrast":m1+" - "+m2,"metric":met,"paired_mean_difference":float(arr.mean()),
                            "paired_sd":float(arr.std(ddof=1)),"ci95_low":float(np.quantile(boots,.025)),
                            "ci95_high":float(np.quantile(boots,.975)),"n_pairs":len(arr)})
    return out


def plot_curves(rows, path):
    fig, axes=plt.subplots(1,2,figsize=(12,4.5))
    for method in sorted({r["method"] for r in rows}):
        for app, start, ax in (("first_C",120,axes[0]),("recurrence_C",208,axes[1])):
            subset=[r for r in rows if r["method"]==method and start<=int(r["step"])<start+48]
            points=[]
            for k in (0,1,2,4,8,16,32):
                vals=[r["preupdate_mse"] for r in subset if int(r["step"])-start==k]
                if vals: points.append((k,np.mean(vals),np.std(vals,ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0))
            if points:
                x,y,e=np.asarray(points).T; ax.plot(x,y,marker="o",label=method); ax.fill_between(x,y-1.96*e,y+1.96*e,alpha=.10)
            ax.axvline(0,color="black",linestyle="--",alpha=.4)
            ax.set_title("First C" if app=="first_C" else "C recurrence"); ax.set_xlabel("C transitions observed"); ax.set_ylabel("Pre-update state MSE"); ax.grid(alpha=.25)
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def near_regime_control(seeds):
    """Controlled Gaussian experiment at sub-noise and noise-scale separation.

    This is an inference limit diagnostic, not environment training data. The
    two theories are fixed and equally likely; only their transition samples
    arrive sequentially. It demonstrates how long evidence needs when
    predictive means are close relative to observation noise.
    """
    out=[]; checkpoints=(1,2,4,8,16,32); sigma=.025
    for ratio in (.20,.60):
        delta=ratio*sigma
        for seed in seeds:
            rng=np.random.default_rng(seed+int(ratio*10000))
            target=int(rng.integers(2)); mu=np.asarray([[0.,0.],[delta,0.]])
            samples=rng.normal(mu[target],sigma,size=(32,2)); logw=np.zeros(2)
            for k,obs in enumerate(samples,1):
                ll=-.5*np.sum(((obs[None]-mu)/sigma)**2,axis=1)
                logw+=ll; p=np.exp(logw-logw.max()); p/=p.sum()
                if k in checkpoints:
                    out.append({"separation_over_noise":ratio,"seed":seed,"observations":k,
                                "label_accuracy":int(np.argmax(p)==target),"target_posterior":float(p[target]),
                                "entropy":float(-np.sum(p*np.log(p.clip(1e-300))))})
    return out


def recurrence_memory_audit(rows, c_target_index):
    """Evaluator-only audit of what predictive candidate memory retrieved."""
    selected=[r for r in rows if r["method"]=="compositional_streaming_revision_memory"]
    out=[]
    for seed in sorted({int(r["seed"]) for r in selected}):
        episode=[r for r in selected if int(r["seed"])==seed]
        first=[r for r in episode if 120<=int(r["step"])<168]
        returning=[r for r in episode if 208<=int(r["step"])<256]
        previously_updated={int(r["selected_candidate"]) for r in first}
        hits=[r for r in returning if int(r["selected_candidate"]) in previously_updated]
        out.append({"seed":seed,
            "first_C_exact_completion_selection_accuracy":float(np.mean([int(r["selected_candidate"])==c_target_index for r in first])),
            "return_C_exact_completion_selection_accuracy":float(np.mean([int(r["selected_candidate"])==c_target_index for r in returning])),
            "posterior_map_label_accuracy_on_return_C":float(np.mean([r["label_based_accuracy"] for r in returning])),
            "retrieval_delay_of_first_C_updated_candidate":int(hits[0]["step"]-208+1) if hits else 48,
            "fraction_return_C_using_first_C_updated_candidate":float(len(hits)/len(returning)),
            "fraction_return_C_selecting_aligned_completion":float(np.mean([int(r["selected_candidate"])==c_target_index for r in returning])),
            "behavioral_map_fit_on_return_C":float(np.mean([r["behavioral_identification_accuracy"] for r in returning])),
            "return_C_preupdate_mse":float(np.mean([r["preupdate_mse"] for r in returning]))})
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=ROOT/"results/rwt/stage2c")
    parser.add_argument("--seeds",default=",".join(map(str,SEEDS))); parser.add_argument("--max-seeds",type=int,default=20)
    args=parser.parse_args(); seeds=tuple(map(int,args.seeds.split(",")))[:args.max_seeds]
    if len(seeds)<20: raise SystemExit("Stage 2C final evaluation requires at least 20 seeds")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    expert,mono=load_models(); factories=prediction_factories(expert,mono)
    thresholds={}; validation_scores={}
    for model in ("composition","experts"):
        threshold,scores=calibrate_threshold(factories[model],3,(3,4,5) if model=="composition" else ())
        thresholds[model]=threshold; validation_scores[model]=scores
    train_data,_=collect("train",list(range(42,54)))
    alignments={"composition":evaluator_alignment(expert,train_data,"composition"),
                "experts":evaluator_alignment(expert,train_data,"experts")}
    print("auditing Stage 2B label/behavior alignment and recurrence cache",flush=True)
    audit_rows,pair_rows,recurrence_rows=audit_stage2b(expert,train_data,seeds)
    write_csv(args.output_dir/"stage2b_audit_trace.csv",audit_rows)
    write_csv(args.output_dir/"stage2b_pairwise_separation.csv",pair_rows)
    write_csv(args.output_dir/"stage2b_recurrence_posterior.csv",recurrence_rows)
    audit_summary=[]
    for policy in ("passive","active"):
        for regime in ("A","B","D","C"):
            sub=[r for r in audit_rows if r["policy"]==policy and r["regime_eval_only"]==regime]
            audit_summary.append({"policy":policy,"regime":regime,"n":len(sub),
                "label_based_accuracy":float(np.nanmean([r["label_based_accuracy"] for r in sub])),
                "behavioral_identification_accuracy":float(np.nanmean([r["behavioral_identification_accuracy"] for r in sub])),
                "map_candidate_mse":float(np.mean([r["map_candidate_mse"] for r in sub])),
                "best_candidate_mse":float(np.mean([r["best_candidate_mse"] for r in sub])),
                "aligned_candidate_mse":float(np.nanmean([r["aligned_candidate_mse"] for r in sub])),
                "target_standardized_residual_rms":float(np.nanmean([r["target_standardized_residual_rms"] for r in sub])),
                "target_coverage_68":float(np.nanmean([r["target_coverage_68"] for r in sub])),
                "target_coverage_95":float(np.nanmean([r["target_coverage_95"] for r in sub]))})
    write_csv(args.output_dir/"stage2b_audit_summary.csv",audit_summary)
    pair_summary=[]
    for policy in ("passive","active"):
        for regime in ("A","B","D","C"):
            sub=[r for r in pair_rows if r["policy"]==policy and r["regime_eval_only"]==regime]
            pair_summary.append({"policy":policy,"regime":regime,
                "mean_pairwise_velocity_distance":float(np.mean([r["mean_pairwise_velocity_distance"] for r in sub])),
                "min_pairwise_velocity_distance":float(np.mean([r["min_pairwise_velocity_distance"] for r in sub])),
                "mean_pairwise_noise_normalized_distance":float(np.mean([r["mean_pairwise_noise_normalized_distance"] for r in sub]))})
    write_csv(args.output_dir/"stage2b_pairwise_summary.csv",pair_summary)
    streams=[collect_stream(seed) for seed in seeds]
    methods=("compositional_frozen","compositional_revision","compositional_revision_memory",
             "compositional_streaming_revision","compositional_streaming_revision_memory",
             "independent_experts_memory","sliding_window","monolithic_full_finetune")
    summaries=[]; traces=[]
    for method in methods:
        print("evaluating",method,flush=True)
        if method.startswith("compositional_streaming"):
            rows,step_rows=evaluate_streaming_adapters(method,streams,factories,alignments)
        else:
            rows,step_rows=evaluate_revision(method,streams,factories,thresholds,alignments,mono)
        summaries.extend(rows); traces.extend(step_rows)
    write_csv(args.output_dir/"episode_metrics.csv",summaries)
    write_csv(args.output_dir/"online_traces.csv",traces)
    write_csv(args.output_dir/"summary.csv",_summary(summaries,("method",),tuple(k for k in summaries[0] if k not in {"method","seed"})))
    write_csv(args.output_dir/"paired_comparisons.csv",paired(summaries,("mse_C","mse_first_C","mse_recurrence_C","mse_A_recurrence","nll_C","recurrence_retrieval_delay","model_recovery_delay_recurrence","false_alarm_rate_known","adaptation_samples_used","online_parameter_budget","compute_seconds")))
    write_csv(args.output_dir/"adaptation_curves.csv",[r for r in traces if r["regime_eval_only"]=="C"])
    write_csv(args.output_dir/"recurrence_memory_audit.csv",
              recurrence_memory_audit(traces,alignments["composition"]["C"]))
    validation_summary=[{"model":m,"n":len(v),"threshold_q995_min_standardized_residual":thresholds[m],
                         "validation_score_mean":float(np.mean(v)),"validation_score_sd":float(np.std(v,ddof=1))} for m,v in validation_scores.items()]
    write_csv(args.output_dir/"mismatch_calibration.csv",validation_summary)
    write_csv(args.output_dir/"near_regime_control.csv",near_regime_control(seeds))
    config={"offline_train_regimes":["A","B","D"],"offline_validation_regimes":["A","B","D"],"C_excluded_from_training_and_calibration":True,
            "online_schedule":[[x.regime,x.duration] for x in SEGMENTS],"evaluation_seeds":seeds,
            "mismatch_threshold_source":"A/B/D validation q=.995 of minimum standardized velocity residual",
            "revision":"fixed Stage2B learned backbone; RLS residual adapter 5x2=10 weights; trigger >=2/3 surprises",
            "window_size":24,"action_norm":MAX_ACTION,"change_schedule_agent_visible":False,
            "near_regime_control":"synthetic Gaussian means separated by 0.20 or 0.60 predictive standard deviations; evaluator-only controlled limit test"}
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    plot_curves(traces,args.output_dir/"stage2c_adaptation.png")
    print("wrote Stage 2C results to",args.output_dir,flush=True)


if __name__=="__main__": main()
