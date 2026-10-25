"""Fresh matched fixed-policy evaluation for Q1 V5 on CPU."""
from __future__ import annotations

import csv
import argparse
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

from q1.datasets import rollout, stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS, apply_program, response, sample_params
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import gaussian_posterior
from q1_v2.inference import composed as v2_composed, rank_programs as v2_rank
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import design as v2_design
from q1_v3.mechanisms import LegacyBoundedOddAction, mean_delta_v as v3_mean
from q1_v3.operators import MechanismOperatorBank
from q1_v3.program_search import score_program_budgets, predict_effect
from q1_v3.task_inference import fit_task_map
from q1.model import ContextDynamicsModel, TransitionNormalizer
from q1.primitives import PrimitiveBank
from q1.evaluate import _direct_context, rank_programs as m4_rank
from q1_v5.model import ProgramModel, fit_theta, joint_map, normal_mixture_shrink, response as v5_response
from q1_v4.stage_contrast import learn_opaque_codes
from q1_v5.joint_refit import joint_refit
from scripts.q1_v3_run import gather_all_background_seeds, tensor_data

OUT = ROOT / "results/q1/q1_v5_fixed_confirmatory_dev"
BASELINES = ("V5", "V2_RETRAINED", "V3", "M4", "POPULATION_MEAN", "IDENTITY_OPERATOR", "DIRECT_CONTEXT")
DT, SIGMA = .05, .0005
BUDGETS = [5, 10, 20, 50, 100]
NTRAIN, NINNER, NDEV = 40, 8, 16
OOD_RANGES = {"background_ood": ((1.22, 1.38), (.60, .72), (.21, .27), (.39, .47))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def bgseeds(n, rng, forbidden):
    out = []
    while len(out) < n:
        value = int(rng.integers(1, 2**31 - 1))
        if value not in forbidden and value not in out:
            out.append(value)
    forbidden.update(out)
    return out


def params_from_seeds(seeds, region):
    return [sample_params(np.random.default_rng(s), region) for s in seeds]


def xy(tr):
    return np.c_[tr.states[:-1], tr.actions], tr.delta_v


def boot(vals, seed=7081, reps=5000):
    x = np.asarray(vals, dtype=float)
    if not len(x): return [None, None]
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(reps, len(x)), replace=True).mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def grouped(rows, field, filt):
    out = {}
    for r in rows:
        if filt(r): out.setdefault(r["background"], []).append(float(r[field]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def paired(a, b, seed):
    keys = sorted(set(a) & set(b)); d = [a[k] - b[k] for k in keys]
    return {"n_backgrounds": len(d), "mean_difference": float(np.mean(d)) if d else None,
            "ci95_background_bootstrap": boot(d, seed)}


def train_v5(train_params, train_seeds, inner_params, inner_seeds, seed):
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    before, pairs, after = [], {i: [] for i in range(4)}, {i: [] for i in range(4)}
    for bi, (p, s) in enumerate(zip(train_params, train_seeds)):
        x, y = xy(rollout(p, n=512, seed=s + 100000, sigma=SIGMA)); before.append((x, y))
        for c in range(4):
            xx, yy = xy(rollout(apply_program(p, (c,)), n=512,
                                 seed=s + 200000 + c * 1009, sigma=SIGMA))
            pairs[c].append((x, y, xx, yy)); after[c].append((xx, yy))
    cfit, thetas, train_loss = _train_shared(before)
    perm = np.random.default_rng(seed + 404).permutation(4)
    p2o = {i: int(perm[i]) for i in range(4)}
    opaque_pairs = {p2o[i]: rows for i, rows in pairs.items()}
    learned = learn_opaque_codes(opaque_pairs, max_degree=5)
    ops = {}
    for physical, opaque in p2o.items():
        mech = learned[opaque].mechanism
        if mech == "action":
            b0 = np.stack([t[:4].reshape(2, 2) for t in thetas])
            b1 = np.stack([fit_theta(*after[physical][i], cfit, guess=thetas[i])[0][:4].reshape(2, 2)
                           for i in range(len(thetas))])
            m = np.concatenate(list(b0), axis=1); target = np.concatenate(list(b1), axis=1)
            ops[opaque] = {"mechanism": "action", "L": target @ np.linalg.pinv(m)}
        else:
            j = ("linear_drag", "quadratic_drag", "position_feedback").index(mech)
            vals = [fit_theta(*after[physical][i], cfit, guess=thetas[i])[0][4+j] - thetas[i][4+j]
                    for i in range(len(thetas))]
            ops[opaque] = {"mechanism": mech, "index": j, "ratio": float(np.exp(np.median(vals)))}
    ops, thetas, refit_steps = joint_refit(cfit, before, after, thetas, ops, p2o, iterations=210)
    mu = np.mean(thetas, axis=0)
    sd = np.maximum(np.std(thetas, axis=0, ddof=1) * 2, [.15, .15, .15, .15, .3, .5, .3])
    model = ProgramModel(cfit, ops, mu, sd)
    prior = {"mean": mu, "sd": sd, "train_loss": train_loss, "joint_refit_iterations": refit_steps,
             "opaque_code_map_for_evaluator": p2o, "operator_mechanisms": {str(k): v["mechanism"] for k,v in ops.items()}}
    prior["operators"] = {str(k): {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk,vv in v.items()} for k,v in ops.items()}
    prior["shared_action_response"] = model_response = cfit
    return model, prior


def _train_shared(before):
    # Reuse the exact observed-data shared-response optimizer from the supplied V5 runner.
    from scripts.q1_v5_experiment import train_shared
    return train_shared(before)


def load_baselines():
    # Frozen published-checkout checkpoints; all are scored on the exact same new rows.
    v2dir = ROOT / "results/q1/q1_v2_bayesian_operators_dev1"
    import yaml
    v2cfg = yaml.safe_load((v2dir / "config.resolved.yaml").read_text())
    v2ck = torch.load(v2dir / "basis_pretrained.pt", map_location="cpu", weights_only=False)
    v2ck["prior_mean"] = v2ck["prior_mean"].double()
    v2ck["prior_covariance"] = v2ck["prior_covariance"].double()
    basis = PartitionedBasis(tuple(v2cfg["blocks"]), hidden=int(v2cfg["hidden"])).double()
    basis.load_state_dict(v2ck["basis"]); basis.eval()
    v2bank = AffineOperatorBank(tuple(v2cfg["blocks"])).double()
    v2op = torch.load(v2dir / "operator_seed4201.pt", map_location="cpu", weights_only=False)
    v2bank.load_state_dict(v2op["state_dict"]); v2bank.eval()
    v2map = {int(k): int(v) for k,v in v2op["opaque_code_map"].items()}
    v3dir = ROOT / "results/q1/q1_v3_mechanism_repair_dev1"
    v3ck = torch.load(v3dir / "v3_primary.pt", map_location="cpu", weights_only=False)
    v3action = LegacyBoundedOddAction().double(); v3action.load_state_dict(v3ck["action_state"]); v3action.eval()
    v3bank = MechanismOperatorBank(dtype=torch.float64)
    v3op = torch.load(v3dir / "operator_seed4201_sparse.pt", map_location="cpu", weights_only=False)
    v3bank.load_state_dict(v3op["state_dict"]); v3bank.eval()
    v3map = {int(k): int(v) for k,v in v3op["opaque_map"].items()}
    m2ck = torch.load(ROOT / "results/q1/cpu_confirmatory_impl/m2/context_model.pt", map_location="cpu", weights_only=False)
    mc = m2ck["config"]["model"]
    m2 = ContextDynamicsModel(int(mc["context_dim"]), int(mc["hidden_dim"]), depth=int(mc.get("depth", 2)), dropout=float(mc.get("dropout", 0.0)))
    m2.load_state_dict(m2ck["model"]); m2.float(); m2.eval()
    normalizer = TransitionNormalizer.from_dict(m2ck["normalizer"])
    m4ck = torch.load(ROOT / "results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt", map_location="cpu", weights_only=False)
    m4 = PrimitiveBank(context_dim=int(mc["context_dim"]))
    m4.load_state_dict(m4ck["primitives"]); m4.float(); m4.eval()
    m4map = {int(k): int(v) for k,v in m4ck["group_to_code"].items()}
    return {"v2": (basis, v2bank, v2ck, v2map),
            "v3": (v3action, v3bank, v3ck, v3map),
            "m2": (m2, normalizer, m2ck), "m4": (m4, m4map, float(m4ck["length_penalty"]))}


def eval_v2(x0,y0,xa,ya,xq,bank,basis,ck,code_map,n):
    programs = [tuple(code_map[c] for c in p) for p in PROGRAMS]
    prior = gaussian_posterior(v2_design(basis, torch.as_tensor(x0)), torch.as_tensor(y0),
                               ck["prior_mean"], ck["prior_covariance"], float(ck["sigma2"]))
    ranked = v2_rank(bank, prior, programs, v2_design(basis, torch.as_tensor(xa[:n])),
                     torch.as_tensor(ya[:n]), float(ck["sigma2"]), 0.)
    pred = tuple(next(k for k,v in code_map.items() if v == c) for c in ranked[0][0])
    mat, bias = v2_composed(bank, ranked[0][0])
    qmean = v2_design(basis, torch.as_tensor(xq)) @ (mat @ prior.mean + bias)
    return pred, qmean.detach().numpy().reshape(-1, 2)


def main():
    global OUT, NTRAIN, NINNER, NDEV
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(OUT));ap.add_argument('--seed',type=int,default=20261031)
    ap.add_argument('--train-backgrounds',type=int,default=NTRAIN);ap.add_argument('--inner-backgrounds',type=int,default=NINNER)
    ap.add_argument('--development-backgrounds',type=int,default=NDEV);args=ap.parse_args()
    OUT=Path(args.output);NTRAIN=args.train_backgrounds;NINNER=args.inner_backgrounds;NDEV=args.development_backgrounds
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError(f"will not overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    started = time.time(); seed = args.seed
    forbidden, audit_sources = gather_all_background_seeds()
    rng = np.random.default_rng(seed)
    tr_s = bgseeds(NTRAIN, rng, forbidden); in_s = bgseeds(NINNER, rng, forbidden)
    id_s = bgseeds(NDEV, rng, forbidden); od_s = bgseeds(NDEV, rng, forbidden)
    train_p = params_from_seeds(tr_s, "train"); inner_p = params_from_seeds(in_s, "train")
    id_p = params_from_seeds(id_s, "train"); od_p = params_from_seeds(od_s, "test_background_ood")
    split = {name: {"seeds": ss, "backgrounds": [{"B": p.B.tolist(), "d": p.d, "b": p.b, "kappa": p.kappa} for p in pp]}
             for name, ss, pp in (("train",tr_s,train_p),("inner",in_s,inner_p),("development_idlike",id_s,id_p),("development_background_ood",od_s,od_p))}
    config = {"protocol": "Q1_FIXED_POLICY", "seed": seed, "dt": DT, "sigma": SIGMA, "old_support": 96,
              "adaptation_transitions": 100, "query_transitions": 256, "budgets": BUDGETS,
              "train_backgrounds": NTRAIN, "inner_backgrounds": NINNER, "development_backgrounds_per_regime": NDEV,
              "programs": [list(p) for p in PROGRAMS], "V5_rule_selection_metric": "inner N=100 functional-class accuracy, tie-break by independent-query MSE; one rule across regimes",
              "acceptance": {"u4_positive_both": True, "u4_positive_paired_bootstrap_ci_vs_population_both": True,
                             "effect_noninferiority_margin_u1_u3": .02, "ordered_pair_accuracy_minimum": .80,
                             "development_class_accuracy_must_beat_matched_v2_both_regimes": True},
              "action_policy": "q1.datasets.stage_pair default sinusoidal-with-jitter, unchanged", "protected_test_used": False}
    (OUT/"config.resolved.json").write_text(json.dumps(config, indent=2))
    (OUT/"split_manifest.json").write_text(json.dumps({"seed":seed,"sets":split,"all_seed_sets_disjoint":True,"protected_test_used":False,"collision_audit_sources":audit_sources},indent=2))
    model, prior = train_v5(train_p,tr_s,inner_p,in_s,seed)
    codes = [tuple(prior["opaque_code_map_for_evaluator"][c] for c in p) for p in PROGRAMS]
    # Inner-only choice of a single V5 estimator; ID-like training distribution only.
    inner_score = {"old_map": {"class": [], "mse": []}, "old_plus_eb": {"class": [], "mse": []}}
    for bi,(p,s) in enumerate(zip(inner_p,in_s)):
        classes,_=functional_classes(p,dt=DT)
        for pi,program in enumerate(PROGRAMS):
            old,adapt,query=stage_pair(p,program,seed=s+300000+pi*7919,n_support=96,n_query=256,dt=DT,sigma=SIGMA)
            x0,y0=xy(old);xa,ya=xy(adapt);xq,yq=xy(query)
            theta,cov,_=model.old_posterior(x0,y0)
            ests={"old_map":theta.copy()}
            b, se = np.exp(theta[5]), np.exp(theta[5])*np.sqrt(max(cov[5,5],0.))
            eb,_=normal_mixture_shrink(float(b),float(se),float(np.exp(prior["mean"][5])),float(prior["sd"][5]))
            t=theta.copy();t[5]=np.log(max(eb,1e-9));ests["old_plus_eb"]=t
            for name,t0 in ests.items():
                scores=model.score_candidates(xa[:100],ya[:100],t0,cov,codes,[100])[100]
                opaque=scores[0][1];pred=tuple(next(k for k,v in prior["opaque_code_map_for_evaluator"].items() if v==c) for c in opaque)
                cls=int(classes[PROGRAMS.index(pred)]==classes[pi])
                theta_pred=model.transform(t0,opaque)
                mse=float(np.mean((v5_response(xq,theta_pred,model.c)-yq)**2))
                inner_score[name]["class"].append(cls);inner_score[name]["mse"].append(mse)
    chosen=max(inner_score,key=lambda k:(np.mean(inner_score[k]["class"]),-np.mean(inner_score[k]["mse"])))
    prior["inner_selection"]={k:{"class_accuracy":float(np.mean(v["class"])),"query_mse":float(np.mean(v["mse"]))} for k,v in inner_score.items()}
    prior["selected_deployable_rule"]=chosen
    prior["shared_action_response"] = {k: np.asarray(v).tolist() for k,v in prior["shared_action_response"].items()}
    (OUT/"training_records.json").write_text(json.dumps({"v5_training":{k:(v if isinstance(v,(int,float,str,list,dict)) else np.asarray(v).tolist()) for k,v in prior.items()},"inner_selection_metric":config["V5_rule_selection_metric"]},indent=2))
    baselines=load_baselines(); per_rows=[]; effect_rows=[]
    regimes=(("IDLIKE","development_idlike",id_p,id_s),("BACKGROUND_OOD","development_background_ood",od_p,od_s))
    for regime,splitname,params_list,seeds in regimes:
      for bi,(p,bseed) in enumerate(zip(params_list,seeds)):
        bg=f"{regime}-{bi:02d}";classes,_=functional_classes(p,dt=DT)
        # Fit all old-law models once per background, then reuse across 21 matched programs.
        v5_rows=[];v2rows=[];v3rows=[];m4rows=[]
        for pi,program in enumerate(PROGRAMS):
            old,adapt,query=stage_pair(p,program,seed=int(bseed)+500000+pi*7919,n_support=96,n_query=256,dt=DT,sigma=SIGMA)
            x0,y0=xy(old);xa,ya=xy(adapt);xq,yq=xy(query)
            theta,cov,_=model.old_posterior(x0,y0)
            if chosen=="old_plus_eb":
                b,se=np.exp(theta[5]),np.exp(theta[5])*np.sqrt(max(cov[5,5],0.));eb,_=normal_mixture_shrink(float(b),float(se),float(np.exp(prior["mean"][5])),float(prior["sd"][5]));theta[5]=np.log(max(eb,1e-9))
            v3action,v3bank,v3ck,v3map=baselines["v3"]
            v3old=fit_task_map(torch.as_tensor(x0),torch.as_tensor(y0),v3action,v3ck["prior_mean"],v3ck["prior_covariance"],dt=DT,sigma=SIGMA,seed=int(bseed)+pi,starts=1,maxiter=100)
            m2,normalizer,m2ck=baselines["m2"]
            support=np.c_[old.states[:-1],old.actions,old.delta_v].astype(np.float32)
            st=torch.as_tensor(support,dtype=torch.float32)
            st=torch.cat((normalizer.normalize_x(st[:,:6]),normalizer.normalize_dv(st[:,6:])),dim=-1)[None]
            with torch.no_grad(): context=m2.encode(st)
            v2basis,v2bank,v2ck,v2map=baselines["v2"]
            # V3 old task law is estimated only from the same 96 observed support rows.
            v3op_programs=[tuple(v3map[c] for c in pr) for pr in PROGRAMS]
            for n in BUDGETS:
                # V5 selected rule
                sc=model.score_candidates(xa[:n],ya[:n],theta,cov,codes,[n])[n]
                vp=sc[0][1];predv5=tuple(next(k for k,v in prior["opaque_code_map_for_evaluator"].items() if v==c) for c in vp)
                qpred=v5_response(xq,model.transform(theta,vp),model.c)
                per_rows.append({"regime":regime,"background":bg,"true_program":json.dumps(program),"N":n,"model":"V5","predicted_program":json.dumps(predv5),"raw_correct":int(predv5==program),"class_correct":int(classes[PROGRAMS.index(predv5)]==classes[pi]),"independent_query_mse":float(np.mean((qpred-yq)**2))})
                # Baseline V2 posterior and exact composed operator search.
                pred2,q2=eval_v2(x0,y0,xa,ya,xq,v2bank,v2basis,v2ck,v2map,n)
                per_rows.append({"regime":regime,"background":bg,"true_program":json.dumps(program),"N":n,"model":"V2_RETRAINED","predicted_program":json.dumps(pred2),"raw_correct":int(pred2==program),"class_correct":int(classes[PROGRAMS.index(pred2)]==classes[pi]),"independent_query_mse":float(np.mean((q2-yq)**2))})
                # V3 marginal-score baseline from its frozen original action/operator checkpoints.
                theta3=v3old.theta.detach();cov3=v3old.covariance.detach()
                sc3=score_program_budgets(torch.as_tensor(xa,dtype=torch.float64),torch.as_tensor(ya,dtype=torch.float64),theta3,cov3,v3bank,v3action,v3op_programs,[n],dt=DT,sigma=SIGMA)[n][0]
                pred3=tuple(next(k for k,v in v3map.items() if v==c) for c in sc3["program"])
                q3=v3_mean(torch.as_tensor(xq,dtype=torch.float64),sc3["theta_mean"],v3action,DT).detach().numpy()
                per_rows.append({"regime":regime,"background":bg,"true_program":json.dumps(program),"N":n,"model":"V3","predicted_program":json.dumps(pred3),"raw_correct":int(pred3==program),"class_correct":int(classes[PROGRAMS.index(pred3)]==classes[pi]),"independent_query_mse":float(np.mean((q3-yq)**2))})
                # M4 learned latent operators on the identical support/adaptation/query rows.
                m4bank,m4map,m4pen=baselines["m4"]
                tr=np.c_[xa[:n],ya[:n]].astype(np.float32)
                programs4=[tuple(m4map[c] for c in pr) for pr in PROGRAMS]
                xt=torch.as_tensor(tr[:,:6],dtype=torch.float32);yt=torch.as_tensor(tr[:,6:],dtype=torch.float32)
                xn,yn=normalizer.normalize_x(xt),normalizer.normalize_dv(yt)
                costs=[]
                with torch.no_grad():
                    for pr4 in programs4:
                        z=m4bank.apply_program(context,pr4)
                        pred=normalizer.denormalize_dv(m2.executor(xn,z))
                        nll=.5*(np.log(2*np.pi*float(m2ck["config"].get("predictive_variance",1e-4)))+
                                (pred-normalizer.denormalize_dv(yn)).square()/float(m2ck["config"].get("predictive_variance",1e-4)))
                        costs.append(float(nll.sum())+m4pen*len(pr4))
                best4=int(np.argmin(costs));ordered=programs4[best4]
                pred4=tuple(next(k for k,v in m4map.items() if v==c) for c in ordered)
                qxt=torch.as_tensor(xq[:,:6],dtype=torch.float32);latent=m4bank.apply_program(context,tuple(m4map[c] for c in pred4))
                with torch.no_grad():q4=normalizer.denormalize_dv(m2.executor(normalizer.normalize_x(qxt),latent)).numpy()
                per_rows.append({"regime":regime,"background":bg,"true_program":json.dumps(program),"N":n,"model":"M4","predicted_program":json.dumps(pred4),"raw_correct":int(pred4==program),"class_correct":int(classes[PROGRAMS.index(pred4)]==classes[pi]),"independent_query_mse":float(np.mean((q4-yq)**2))})
                # Population-mean and identity-operator fixed-policy references.
                for name,base_model in (("POPULATION_MEAN",ProgramModel(model.c,model.ops,prior["mean"],prior["sd"])),("IDENTITY_OPERATOR",ProgramModel(model.c,{k:{"mechanism":"action","L":np.eye(2)} for k in range(4)}))):
                    base_theta=prior["mean"] if name=="POPULATION_MEAN" else theta
                    ss=base_model.score_candidates(xa[:n],ya[:n],base_theta,cov,codes,[n])[n]
                    pp=tuple(next(k for k,v in prior["opaque_code_map_for_evaluator"].items() if v==c) for c in ss[0][1])
                    qq=v5_response(xq,base_model.transform(base_theta,ss[0][1]),model.c)
                    per_rows.append({"regime":regime,"background":bg,"true_program":json.dumps(program),"N":n,"model":name,"predicted_program":json.dumps(pp),"raw_correct":int(pp==program),"class_correct":int(classes[PROGRAMS.index(pp)]==classes[pi]),"independent_query_mse":float(np.mean((qq-yq)**2))})
            # U1-U4 effect diagnostics on the independent query, allowed oracle code only here.
            if len(program)==1:
                code=program[0];actual=response(apply_program(p,program),xq,dt=DT)-response(p,xq,dt=DT);energy=float(np.mean(actual**2))
                # V5 selected model and population / zero-effect.
                opaque=prior["opaque_code_map_for_evaluator"][code]
                d5=v5_response(xq,model.transform(theta,(opaque,)),model.c)-v5_response(xq,theta,model.c)
                for name,delta in (("V5",d5),("POPULATION_MEAN",v5_response(xq,model.transform(prior["mean"],(opaque,)),model.c)-v5_response(xq,prior["mean"],model.c)),("IDENTITY_OPERATOR",np.zeros_like(actual))):
                    mse=float(np.mean((delta-actual)**2));effect_rows.append({"regime":regime,"background":bg,"code":code+1,"model":name,"skill":1-mse/max(energy,1e-30),"effect_mse":mse,"effect_energy":energy})
                # V2 function-space effect from same old posterior.
                p2=gaussian_posterior(v2_design(v2basis,torch.as_tensor(x0)),torch.as_tensor(y0),v2ck["prior_mean"],v2ck["prior_covariance"],float(v2ck["sigma2"]))
                co=v2map[code]
                transformed_mean=v2bank.apply(co,p2.mean)
                d2=(v2_design(v2basis,torch.as_tensor(xq))@(transformed_mean-p2.mean)).detach().numpy().reshape(-1,2)
                mse=float(np.mean((d2-actual)**2));effect_rows.append({"regime":regime,"background":bg,"code":code+1,"model":"V2_RETRAINED","skill":1-mse/max(energy,1e-30),"effect_mse":mse,"effect_energy":energy})
                # V3 effect skill.
                delta3=predict_effect(torch.as_tensor(xq,dtype=torch.float64),v3old.theta,v3bank,v3map[code],v3action,dt=DT).detach().numpy()
                mse=float(np.mean((delta3-actual)**2));effect_rows.append({"regime":regime,"background":bg,"code":code+1,"model":"V3","skill":1-mse/max(energy,1e-30),"effect_mse":mse,"effect_energy":energy})
                # M4 effect skill.
                m4bank,m4map,_=baselines["m4"]; base=m2.executor(normalizer.normalize_x(torch.as_tensor(xq[:,:6],dtype=torch.float32)),context)
                transformed=m2.executor(normalizer.normalize_x(torch.as_tensor(xq[:,:6],dtype=torch.float32)),m4bank.apply_program(context,(m4map[code],)))
                d4=(normalizer.denormalize_dv(transformed)-normalizer.denormalize_dv(base)).detach().numpy()
                mse=float(np.mean((d4-actual)**2));effect_rows.append({"regime":regime,"background":bg,"code":code+1,"model":"M4","skill":1-mse/max(energy,1e-30),"effect_mse":mse,"effect_energy":energy})
            if (pi+1)%7==0: print(f"{regime} background {bi+1}/{NDEV} program {pi+1}/21",flush=True)
        # Direct-context M2 on same background/program cases; compute once per case at N=100.
        # Omitted from class rows by definition; measured in separate query table below.
    # Build direct-context query rows in same case loop (replay exact deterministic seeds).
    direct_rows=[]
    m2,normalizer,m2ck=baselines["m2"]
    for regime,_,params_list,seeds in regimes:
      for bi,(p,bseed) in enumerate(zip(params_list,seeds)):
        bg=f"{regime}-{bi:02d}"
        for pi,program in enumerate(PROGRAMS):
            old,adapt,query=stage_pair(p,program,seed=int(bseed)+500000+pi*7919,n_support=96,n_query=256,dt=DT,sigma=SIGMA)
            support=np.c_[old.states[:-1],old.actions,old.delta_v].astype(np.float32);st=torch.as_tensor(support,dtype=torch.float32)
            st=torch.cat((normalizer.normalize_x(st[:,:6]),normalizer.normalize_dv(st[:,6:])),dim=-1)[None]
            with torch.no_grad():ctx=m2.encode(st)
            x,y=xy(adapt);trans=np.c_[x,y].astype(np.float32)
            xq,yq=xy(query);qt=torch.as_tensor(xq[:,:6],dtype=torch.float32)
            for n in BUDGETS:
                direct=_direct_context(ctx,trans[:n],m2,normalizer,torch.device("cpu"),steps=40,regularization=1e-3)
                with torch.no_grad(): pred=normalizer.denormalize_dv(m2.executor(normalizer.normalize_x(qt),direct)).detach().numpy()
                direct_rows.append({"regime":regime,"background":bg,"true_program":json.dumps(program),"N":n,"model":"DIRECT_CONTEXT","independent_query_mse":float(np.mean((pred-yq)**2))})
    save(OUT/"per_background_per_program_metrics.csv",per_rows);save(OUT/"effect_per_background.csv",effect_rows);save(OUT/"direct_context_per_background.csv",direct_rows)
    prog_summary=[]
    summary={}; comparisons={}
    for reg in ("IDLIKE","BACKGROUND_OOD"):
      for n in BUDGETS:
        for model_name in ("V5","V2_RETRAINED","V3","M4","POPULATION_MEAN","IDENTITY_OPERATOR"):
            cell=[r for r in per_rows if r["regime"]==reg and r["N"]==n and r["model"]==model_name]
            by=grouped(cell,"class_correct",lambda r:True)
            mse_by=grouped(cell,"independent_query_mse",lambda r:True)
            summary[f"{reg}|{model_name}|N={n}"]={"functional_class_accuracy":float(np.mean(list(by.values()))),"class_ci95_background":boot(list(by.values()),seed+1+n),"n_backgrounds":len(by),"independent_query_mse":float(np.mean(list(mse_by.values()))),"query_mse_ci95_background":boot(list(mse_by.values()),seed+2+n)}
            for program in PROGRAMS:
                pc=[r for r in cell if tuple(json.loads(r["true_program"]))==program]
                if not pc: continue
                cb=grouped(pc,"class_correct",lambda r:True);rb=grouped(pc,"raw_correct",lambda r:True);qb=grouped(pc,"independent_query_mse",lambda r:True)
                prog_summary.append({"regime":reg,"model":model_name,"N":n,"true_program":json.dumps(program),
                    "functional_class_accuracy":float(np.mean(list(cb.values()))),
                    "raw_program_accuracy":float(np.mean(list(rb.values()))),
                    "independent_query_mse":float(np.mean(list(qb.values()))),
                    "class_ci95_background":json.dumps(boot(list(cb.values()),seed+11+n)),
                    "raw_ci95_background":json.dumps(boot(list(rb.values()),seed+12+n)),
                    "query_ci95_background":json.dumps(boot(list(qb.values()),seed+13+n))})
      for model_name in ("V5","V2_RETRAINED","V3","M4","POPULATION_MEAN","IDENTITY_OPERATOR"):
        for code in range(1,5):
            e=[r for r in effect_rows if r["regime"]==reg and r["model"]==model_name and r["code"]==code]
            by=grouped(e,"skill",lambda r:True)
            if by: summary[f"{reg}|{model_name}|U{code}"]={"mean_background_skill":float(np.mean(list(by.values()))),"ci95":boot(list(by.values()),seed+code),"n_backgrounds":len(by),"ratio_of_means_skill":1-float(np.mean([r["effect_mse"] for r in e]))/max(float(np.mean([r["effect_energy"] for r in e])),1e-30)}
      for model_name in ("V5","V2_RETRAINED","V3","M4"):
        v5=grouped([r for r in effect_rows if r["regime"]==reg and r["model"]==model_name and r["code"]==4],"skill",lambda r:True)
        pop=grouped([r for r in effect_rows if r["regime"]==reg and r["model"]=="POPULATION_MEAN" and r["code"]==4],"skill",lambda r:True)
        comparisons[f"{reg}|{model_name}|U4_minus_population"]=paired(v5,pop,seed+99)
      for pair in ((0,1),(1,0)):
        for model_name in ("V5","V2_RETRAINED","V3","M4"):
            cell=[r for r in per_rows if r["regime"]==reg and r["N"]==100 and r["model"]==model_name and tuple(json.loads(r["true_program"]))==pair]
            by=grouped(cell,"class_correct",lambda r:True)
            summary[f"{reg}|{model_name}|order={pair}"]={"class_accuracy":float(np.mean(list(by.values()))),"ci95":boot(list(by.values()),seed+33)}
    save(OUT/"per_program_summary.csv",prog_summary)
    direct_summary={}
    for reg in ("IDLIKE","BACKGROUND_OOD"):
        cell=[r for r in direct_rows if r["regime"]==reg]
        by=grouped(cell,"independent_query_mse",lambda r:True)
        direct_summary[reg]={"mean_query_mse":float(np.mean(list(by.values()))),"ci95":boot(list(by.values()),seed+51)}
    selected_rule=prior["selected_deployable_rule"]
    class_vs_v2={}; effect_vs_v3={}; ordered_vs_v3={}
    for reg in ("IDLIKE","BACKGROUND_OOD"):
      for n in BUDGETS:
        v5_by=grouped([r for r in per_rows if r["regime"]==reg and r["N"]==n and r["model"]=="V5"],"class_correct",lambda r:True)
        v2_by=grouped([r for r in per_rows if r["regime"]==reg and r["N"]==n and r["model"]=="V2_RETRAINED"],"class_correct",lambda r:True)
        class_vs_v2[f"{reg}|N={n}"]=paired(v5_by,v2_by,seed+n)
      for code in range(1,4):
        v5_by=grouped([r for r in effect_rows if r["regime"]==reg and r["model"]=="V5" and r["code"]==code],"skill",lambda r:True)
        v3_by=grouped([r for r in effect_rows if r["regime"]==reg and r["model"]=="V3" and r["code"]==code],"skill",lambda r:True)
        effect_vs_v3[f"{reg}|U{code}"]=paired(v5_by,v3_by,seed+code)
      for pair in ((0,1),(1,0)):
        cell=lambda m:[r for r in per_rows if r["regime"]==reg and r["N"]==100 and r["model"]==m and tuple(json.loads(r["true_program"]))==pair]
        a=grouped(cell("V5"),"class_correct",lambda r:True);b=grouped(cell("V3"),"class_correct",lambda r:True)
        ordered_vs_v3[f"{reg}|{pair}"]=paired(a,b,seed+len(pair))
    u4_pass=all(summary.get(f"{r}|V5|U4",{}).get("mean_background_skill",-np.inf)>0 and comparisons[f"{r}|V5|U4_minus_population"]["ci95_background_bootstrap"][0]>0 for r in ("IDLIKE","BACKGROUND_OOD"))
    u3u4_positive=all(summary.get(f"{r}|V5|U{c}",{}).get("mean_background_skill",-np.inf)>0 for r in ("IDLIKE","BACKGROUND_OOD") for c in (3,4))
    u1u3_ni=all(effect_vs_v3[f"{r}|U{c}"]["ci95_background_bootstrap"][0]>=-.02 for r in ("IDLIKE","BACKGROUND_OOD") for c in range(1,4))
    ordered_min=all(summary.get(f"{r}|V5|order={p}",{}).get("class_accuracy",0)>=.80 for r in ("IDLIKE","BACKGROUND_OOD") for p in ((0,1),(1,0)))
    ordered_ni=all(ordered_vs_v3[f"{r}|{p}"]["ci95_background_bootstrap"][0]>=-.02 for r in ("IDLIKE","BACKGROUND_OOD") for p in ((0,1),(1,0)))
    class_beat_v2=all(class_vs_v2[f"{r}|N=100"]["ci95_background_bootstrap"][0]>0 for r in ("IDLIKE","BACKGROUND_OOD"))
    gates={"u4_positive_and_beats_population_both_regimes":u4_pass,"u3_u4_positive_both_regimes":u3u4_positive,
           "u1_u3_effect_noninferior_to_v3_margin_0.02":u1u3_ni,"ordered_pair_accuracy_at_least_0.80":ordered_min,
           "ordered_pair_class_noninferior_to_v3_margin_0.02":ordered_ni,"class_accuracy_beats_v2_both_regimes":class_beat_v2}
    comparisons.update({f"V5_minus_V2_class|{k}":v for k,v in class_vs_v2.items()})
    comparisons.update({f"V5_minus_V3_effect|{k}":v for k,v in effect_vs_v3.items()})
    comparisons.update({f"V5_minus_V3_order|{k}":v for k,v in ordered_vs_v3.items()})
    decision={"protocol":"Q1_FIXED_POLICY","selected_rule":selected_rule,"selected_from_inner_only":True,
              "v5_fixed_gate":"PASS" if all(gates.values()) else "FAIL","gates":gates,"u4_gate_pass":u4_pass,
              "notes":["All baseline rows use the same generated old/adaptation/query observations.","V2/V3/M4 are frozen checkpoints from repository training outputs; evaluation backgrounds and trajectories are new.","Population mean and identity are V5-aligned reference baselines; direct-context is prediction-only and has no class estimate.","Protected test split not accessed."]}
    (OUT/"summary.json").write_text(json.dumps({"metrics":summary,"paired_comparisons":comparisons,"direct_context":direct_summary,"selection_inner":prior["inner_selection"],"decision":decision},indent=2))
    (OUT/"DECISION.md").write_text("# Q1 V5 fixed-policy confirmatory decision\n\n"+json.dumps(decision,indent=2)+"\n\nFull rows and bootstrap summaries are in `per_background_per_program_metrics.csv`, `effect_per_background.csv`, and `summary.json`.\n")
    (OUT/"provenance.json").write_text(json.dumps({"branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),"start_commit":"43245aee373ddd1061406a8c22f17f7ba686e30d","head_before_run":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"device":"cpu","seconds":time.time()-started,"protected_test_used":False,"source_sha256":{str(Path(__file__).relative_to(ROOT)):sha(__file__)},"baseline_checkpoints":{"v2":sha(ROOT/"results/q1/q1_v2_bayesian_operators_dev1/operator_seed4201.pt"),"v3":sha(ROOT/"results/q1/q1_v3_mechanism_repair_dev1/operator_seed4201_sparse.pt"),"m4":sha(ROOT/"results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt")}},indent=2))
    print(json.dumps(decision,indent=2))


if __name__ == "__main__": main()
