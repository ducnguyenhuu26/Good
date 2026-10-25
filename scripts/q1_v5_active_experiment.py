"""Separate CPU development study for Q1_ACTIVE_EXCITATION."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
import torch

from q1.datasets import rollout
from q1.exploration import action_at
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS,apply_program,response, sample_params,transition
from q1_v5.active_design import FisherActionPolicy,full_response_jacobian
from q1_v5.identifiability import efficient_information
from q1_v5.model import DT,SIGMA,ProgramModel,fit_theta,joint_map,normal_mixture_shrink,response as v5_response
from scripts.q1_v3_run import gather_all_background_seeds

OUT=ROOT/"results/q1/q1_v5_active_excitation_dev"
BUDGETS=[5,10,20,50,100];POLICIES=("FIXED_SINUSOIDAL","RANDOM_UNIFORM","ACTIVE_I_EFF_LOGB","UNCERTAINTY_D_OPTIMAL")


def xy(tr):return np.c_[tr.states[:-1],tr.actions],tr.delta_v
def save(path,rows):
    if rows:
        with Path(path).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def boot(vals,seed,reps=5000):
    a=np.asarray(vals,dtype=float)
    if not len(a):return [None,None]
    b=np.random.default_rng(seed).choice(a,size=(reps,len(a)),replace=True).mean(1)
    return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]
def bgmeans(rows,key,value,where):
    d={}
    for r in rows:
        if where(r):d.setdefault(r[key],[]).append(float(r[value]))
    return {k:float(np.mean(v)) for k,v in d.items()}
def paired(a,b,seed):
    ks=sorted(set(a)&set(b));v=[a[k]-b[k] for k in ks]
    return {"n_backgrounds":len(v),"mean_difference":float(np.mean(v)) if v else None,"ci95_background_bootstrap":boot(v,seed)}


def load_v5():
    fixed=ROOT/"results/q1/q1_v5_fixed_confirmatory_dev"
    records=json.loads((fixed/"training_records.json").read_text(encoding="utf-8"))["v5_training"]
    c=records["shared_action_response"]
    ops={int(k):{kk:(np.asarray(vv,dtype=float) if kk=="L" else vv) for kk,vv in val.items()} for k,val in records["operators"].items()}
    mean=np.asarray(records["mean"],dtype=float);sd=np.asarray(records["sd"],dtype=float)
    model=ProgramModel(c,ops,mean,sd)
    return model,records,mean,sd


def generate_adaptation(before,actual_program,seed,old,theta,c,policy_name,selector):
    """Simulate a new protocol trajectory. The action selector receives no oracle arguments."""
    params_after=apply_program(before,actual_program)
    action_rng=np.random.default_rng(seed+104730);noise_rng=np.random.default_rng(seed+209760)
    states=[old.states[-1].copy()];actions=[];history_x=np.c_[old.states[:-1],old.actions].copy()
    old_j=full_response_jacobian(history_x,theta,c);info=old_j.T@old_j/(SIGMA**2)
    candidates=0;violations=0
    for t in range(100):
        state=states[-1]
        if policy_name=="FIXED_SINUSOIDAL":
            action=action_at(t,action_rng,1.0,.85,.15)
        elif policy_name=="RANDOM_UNIFORM":
            action=action_rng.uniform(-1,1,2)
        else:
            action=selector.choose(state,history_x,theta,c,history_information=info)
            candidates+=len(selector.last_scores)
        action=np.asarray(action,dtype=float)
        if np.max(np.abs(action))>1+1e-12:violations+=1
        x=np.r_[state,action]
        next_state=transition(state,action,params_after,dt=DT,sigma=SIGMA,rng=noise_rng)
        if np.linalg.norm(next_state)>100:raise FloatingPointError("active protocol state exceeded safety envelope")
        j=full_response_jacobian(x[None,:],theta,c);info+=j.T@j/(SIGMA**2)
        actions.append(action);history_x=np.vstack((history_x,x));states.append(next_state)
    from q1.datasets import Trajectory
    adapt=Trajectory(np.asarray(states),np.asarray(actions))
    query=rollout(params_after,n=256,seed=seed+2,dt=DT,sigma=SIGMA)
    return adapt,query,info,candidates,violations


def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument("--output",default=str(OUT));ap.add_argument("--seed",type=int,default=20261032);ap.add_argument("--backgrounds-per-regime",type=int,default=4)
    args=ap.parse_args();OUT=Path(args.output)
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(f"will not overwrite {OUT}")
    OUT.mkdir(parents=True,exist_ok=True);started=time.time();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    model,training,mean,sd=load_v5();rule=json.loads((ROOT/"results/q1/q1_v5_fixed_confirmatory_dev/summary.json").read_text(encoding="utf-8"))["decision"]["selected_rule"]
    forbidden,sources=gather_all_background_seeds();rng=np.random.default_rng(args.seed)
    seeds={}
    for key in ("development_idlike","development_background_ood"):
        count=args.backgrounds_per_regime
        ss=[]
        while len(ss)<count:
            s=int(rng.integers(1,2**31-1))
            if s not in forbidden and s not in ss:ss.append(s)
        forbidden.update(ss);seeds[key]=ss
    idp=[sample_params(np.random.default_rng(s),"train") for s in seeds["development_idlike"]]
    odp=[sample_params(np.random.default_rng(s),"test_background_ood") for s in seeds["development_background_ood"]]
    manifest={"seed":args.seed,"sets":{k:{"seeds":v} for k,v in seeds.items()},
              "development_backgrounds":{"idlike":[{"B":p.B.tolist(),"d":p.d,"b":p.b,"kappa":p.kappa} for p in idp],"background_ood":[{"B":p.B.tolist(),"d":p.d,"b":p.b,"kappa":p.kappa} for p in odp]},
              "all_development_seeds_disjoint_from_saved_manifests":True,"protected_test_used":False,"seed_audit_sources":sources}
    config={"protocol":"Q1_ACTIVE_EXCITATION","dt":DT,"sigma":SIGMA,"old_support":96,"adaptation_transitions":100,"query_transitions":256,"budgets":BUDGETS,
            "backgrounds_per_regime":args.backgrounds_per_regime,"policies":list(POLICIES),"candidate_action_grid":[-1.,0.,1.],"action_bounds":[-1.,1.],
            "greedy_horizon":2,"active_utility":"log(1 + nuisance-residualized Fisher information for log(b)) after a two-step frozen-model rollout",
            "uncertainty_control":"full twelve-coordinate log-determinant of estimated information matrix",
            "predeclared_active_success_gate":{"N":100,"I_eff_logb_active_minus_sinusoidal_and_random_ci95_lower_gt_zero_both_regimes":True,"U4_skill_positive_both_regimes":True,"U4_skill_active_minus_sinusoidal_ci95_lower_gt_zero_both_regimes":True},
            "frozen_v5_rule":rule,"selector_inputs":["observed current state","old and adaptation observed state/action history","frozen learned theta and action response"],
            "oracle_policy_inputs":False,"protected_test_used":False,"v5_model_source":"Phase B frozen training records; no active-protocol retraining or hyperparameter selection",
            "policy_changes":"adaptation action sequence only; old support, transition counts, simulator and independent query protocol unchanged"}
    (OUT/"config.resolved.json").write_text(json.dumps(config,indent=2));(OUT/"split_manifest.json").write_text(json.dumps(manifest,indent=2))
    (OUT/"training_records.json").write_text((ROOT/"results/q1/q1_v5_fixed_confirmatory_dev/training_records.json").read_text(encoding="utf-8"))
    if rule=="old_plus_eb":
        # One frozen train-derived shrink rule, identical across all active regimes.
        active_rule="old_plus_eb"
    else:active_rule="old_map"
    p2o={int(k):int(v) for k,v in training["opaque_code_map_for_evaluator"].items()};o2p={v:k for k,v in p2o.items()}
    opaque_programs=[tuple(p2o[i] for i in pr) for pr in PROGRAMS]
    rows=[];effects=[];fisher=[];action_rows=[];runtime_by_policy={p:0. for p in POLICIES};candidate_counts={p:0 for p in POLICIES};violations={p:0 for p in POLICIES}
    regimes=(("IDLIKE","development_idlike",idp,seeds["development_idlike"]),("BACKGROUND_OOD","development_background_ood",odp,seeds["development_background_ood"]))
    for regime,_,params_list,case_seeds in regimes:
      for bi,(p,bseed) in enumerate(zip(params_list,case_seeds)):
       bg=f"{regime}-{bi:02d}";classes,_=functional_classes(p,dt=DT)
       for pi,actual_program in enumerate(PROGRAMS):
        case_seed=int(bseed)+600000+pi*7919
        old=rollout(p,n=96,seed=case_seed,dt=DT,sigma=SIGMA);x0,y0=xy(old)
        theta,cov,_=model.old_posterior(x0,y0)
        if active_rule=="old_plus_eb":
            b,se=np.exp(theta[5]),np.exp(theta[5])*np.sqrt(max(cov[5,5],0.));eb,_=normal_mixture_shrink(float(b),float(se),float(np.exp(mean[5])),float(sd[5]));theta[5]=np.log(max(eb,1e-9))
        for policy_name in POLICIES:
            selector=None
            if policy_name=="ACTIVE_I_EFF_LOGB":selector=FisherActionPolicy("efficient_logb",horizon=2,dt=DT,sigma=SIGMA)
            elif policy_name=="UNCERTAINTY_D_OPTIMAL":selector=FisherActionPolicy("d_optimal",horizon=2,dt=DT,sigma=SIGMA)
            t0=time.perf_counter();adapt,query,final_info,candidates,bounds=generate_adaptation(p,actual_program,case_seed,old,theta,model.c,policy_name,selector);runtime_by_policy[policy_name]+=time.perf_counter()-t0;candidate_counts[policy_name]+=candidates;violations[policy_name]+=bounds
            xa,ya=xy(adapt);xq,yq=xy(query);energy=None
            classes,_=functional_classes(p,dt=DT)
            for n in BUDGETS:
                rank=model.score_candidates(xa[:n],ya[:n],theta,cov,opaque_programs,[n])[n]
                pred_opaque=rank[0][1];pred=tuple(o2p[c] for c in pred_opaque)
                # After the frozen old-posterior ranking, refine only the selected candidate.
                # This uses observed transitions and the selected hypothesis, never true code.
                jm,jcov,jloss=joint_map(model,x0,y0,xa[:n],ya[:n],pred_opaque,initial_theta=theta)
                pred_theta=model.transform(jm,pred_opaque)
                qpred=v5_response(xq,pred_theta,model.c)
                correct_class=int(classes[PROGRAMS.index(pred)]==classes[PROGRAMS.index(actual_program)])
                rows.append({"regime":regime,"background":bg,"policy":policy_name,"true_program":json.dumps(actual_program),"predicted_program":json.dumps(pred),"N":n,
                             "raw_program_correct":int(pred==actual_program),"functional_class_correct":correct_class,"independent_query_mse":float(np.mean((qpred-yq)**2)),
                             "b_hat":float(np.exp(jm[5])),"b_true_offline":p.b,"abs_b_error_offline":abs(float(np.exp(jm[5]))-p.b),"joint_map_loss":jloss})
                design=np.vstack((x0,xa[:n]));j=full_response_jacobian(design,theta,model.c);info=efficient_information(j,5,sigma=SIGMA)
                fisher.append({"regime":regime,"background":bg,"policy":policy_name,"true_program":json.dumps(actual_program),"N":n,
                               "I_eff_logb":info["efficient_information"],"local_logb_se":info["crlb_se"],"velocity_rms":float(np.sqrt(np.mean(design[:,2:4]**2))),
                               "velocity_qdrag_cosine":float(np.dot(design[:,2:4].ravel(),(np.linalg.norm(design[:,2:4],axis=1,keepdims=True)*design[:,2:4]).ravel())/
                                   max(np.linalg.norm(design[:,2:4])*np.linalg.norm((np.linalg.norm(design[:,2:4],axis=1,keepdims=True)*design[:,2:4])),1e-30))})
                if actual_program==(3,):
                    true_delta=response(apply_program(p,actual_program),xq,dt=DT)-response(p,xq,dt=DT);eff_energy=float(np.mean(true_delta**2))
                    pred_delta=qpred-v5_response(xq,jm,model.c)
                    pop_delta=v5_response(xq,model.transform(mean,pred_opaque),model.c)-v5_response(xq,mean,model.c)
                    for method,delta in (("ACTIVE_SELECTED_PROGRAM",pred_delta),("POPULATION_MEAN_SELECTED_PROGRAM",pop_delta)):
                        mse=float(np.mean((delta-true_delta)**2));effects.append({"regime":regime,"background":bg,"policy":policy_name,"N":n,"model":method,
                            "effect_skill":1-mse/max(eff_energy,1e-30),"effect_mse":mse,"true_effect_energy":eff_energy,"zero_effect_skill":0.0})
            for t,action in enumerate(adapt.actions):
                action_rows.append({"regime":regime,"background":bg,"true_program":json.dumps(actual_program),"policy":policy_name,"step":t,
                                    "selected_action":json.dumps(action.tolist()),"candidate_count":len(selector.actions) if selector is not None else 0,
                                    "selected_utility":(dict(selector.last_scores).get(tuple(action),None) if selector is not None and t==99 else None)})
        if (pi+1)%7==0:print(f"{regime} background {bi+1}/{len(params_list)} program {pi+1}/21",flush=True)
    save(OUT/"per_background_per_program_metrics.csv",rows);save(OUT/"fisher_per_background.csv",fisher);save(OUT/"u4_effect_per_background.csv",effects);save(OUT/"action_decisions.csv",action_rows)
    summary={};paired_comparisons={}
    for regime in ("IDLIKE","BACKGROUND_OOD"):
      for policy in POLICIES:
       for n in BUDGETS:
        cell=[r for r in rows if r["regime"]==regime and r["policy"]==policy and r["N"]==n]
        for metric in ("functional_class_correct","raw_program_correct","independent_query_mse","abs_b_error_offline"):
            bm=bgmeans(cell,"background",metric,lambda _:True);summary[f"{regime}|{policy}|N={n}|{metric}"]={"mean_background_value":float(np.mean(list(bm.values()))),"ci95_background":boot(list(bm.values()),args.seed+n+3),"n_backgrounds":len(bm)}
       for n in BUDGETS:
        fi=[r for r in fisher if r["regime"]==regime and r["policy"]==policy and r["N"]==n]
        for metric in ("I_eff_logb","local_logb_se","velocity_rms","velocity_qdrag_cosine"):
            bm=bgmeans(fi,"background",metric,lambda _:True);summary[f"{regime}|{policy}|N={n}|{metric}"]={"mean_background_value":float(np.mean(list(bm.values()))),"ci95_background":boot(list(bm.values()),args.seed+n+8),"n_backgrounds":len(bm)}
       for n in BUDGETS:
        ee=[r for r in effects if r["regime"]==regime and r["policy"]==policy and r["N"]==n and r["model"]=="ACTIVE_SELECTED_PROGRAM"]
        if ee:
            bm=bgmeans(ee,"background","effect_skill",lambda _:True);summary[f"{regime}|{policy}|N={n}|U4_effect_skill"]={"mean_background_skill":float(np.mean(list(bm.values()))),"ci95_background":boot(list(bm.values()),args.seed+n+13),"n_backgrounds":len(bm)}
            pm=bgmeans([r for r in effects if r["regime"]==regime and r["policy"]==policy and r["N"]==n and r["model"]=="POPULATION_MEAN_SELECTED_PROGRAM"],"background","effect_skill",lambda _:True)
            paired_comparisons[f"{regime}|{policy}|N={n}|U4_minus_population"]=paired(bm,pm,args.seed+n+15)
    for reg in ("IDLIKE","BACKGROUND_OOD"):
      for policy in POLICIES[1:]:
        for n in BUDGETS:
          for metric in ("functional_class_correct","I_eff_logb","abs_b_error_offline"):
            source=rows if metric=="functional_class_correct" or metric=="abs_b_error_offline" else fisher
            field={"functional_class_correct":"functional_class_correct","abs_b_error_offline":"abs_b_error_offline","I_eff_logb":"I_eff_logb"}[metric]
            fil=lambda r:r["regime"]==reg and r["policy"]==policy and r["N"]==n
            basepol="FIXED_SINUSOIDAL";aa=bgmeans([r for r in source if fil(r)],"background",field,lambda _:True)
            bb=bgmeans([r for r in source if r["regime"]==reg and r["policy"]==basepol and r["N"]==n],"background",field,lambda _:True)
            paired_comparisons[f"{reg}|{policy}_minus_sinusoidal|N={n}|{metric}"]=paired(aa,bb,args.seed+n+22)
          aa=bgmeans([r for r in effects if r["regime"]==reg and r["policy"]==policy and r["N"]==n and r["model"]=="ACTIVE_SELECTED_PROGRAM"],"background","effect_skill",lambda _:True)
          bb=bgmeans([r for r in effects if r["regime"]==reg and r["policy"]=="FIXED_SINUSOIDAL" and r["N"]==n and r["model"]=="ACTIVE_SELECTED_PROGRAM"],"background","effect_skill",lambda _:True)
          paired_comparisons[f"{reg}|{policy}_minus_sinusoidal|N={n}|U4_effect_skill"]=paired(aa,bb,args.seed+n+30)
      for policy in ("RANDOM_UNIFORM","UNCERTAINTY_D_OPTIMAL"):
        for n in BUDGETS:
          for metric,source,field in (("I_eff_logb",fisher,"I_eff_logb"),("functional_class_accuracy",rows,"functional_class_correct"),("abs_b_error_offline",rows,"abs_b_error_offline")):
            aa=bgmeans([r for r in source if r["regime"]==reg and r["policy"]=="ACTIVE_I_EFF_LOGB" and r["N"]==n],"background",field,lambda _:True)
            bb=bgmeans([r for r in source if r["regime"]==reg and r["policy"]==policy and r["N"]==n],"background",field,lambda _:True)
            paired_comparisons[f"{reg}|ACTIVE_I_EFF_LOGB_minus_{policy}|N={n}|{metric}"]=paired(aa,bb,args.seed+n+31)
        for n in BUDGETS:
          aa=bgmeans([r for r in effects if r["regime"]==reg and r["policy"]=="ACTIVE_I_EFF_LOGB" and r["N"]==n and r["model"]=="ACTIVE_SELECTED_PROGRAM"],"background","effect_skill",lambda _:True)
          bb=bgmeans([r for r in effects if r["regime"]==reg and r["policy"]==policy and r["N"]==n and r["model"]=="ACTIVE_SELECTED_PROGRAM"],"background","effect_skill",lambda _:True)
          paired_comparisons[f"{reg}|ACTIVE_I_EFF_LOGB_minus_{policy}|N={n}|U4_effect_skill"]=paired(aa,bb,args.seed+n+32)
    info_pass=all(paired_comparisons[key]["ci95_background_bootstrap"][0]>0
                  for reg in ("IDLIKE","BACKGROUND_OOD")
                  for key in (f"{reg}|ACTIVE_I_EFF_LOGB_minus_sinusoidal|N=100|I_eff_logb",
                              f"{reg}|ACTIVE_I_EFF_LOGB_minus_RANDOM_UNIFORM|N=100|I_eff_logb"))
    u4_positive=all(summary.get(f"{reg}|ACTIVE_I_EFF_LOGB|N=100|U4_effect_skill",{}).get("ci95_background",[None,None])[0]>0
                    for reg in ("IDLIKE","BACKGROUND_OOD"))
    u4_beats_fixed=all(paired_comparisons[f"{reg}|ACTIVE_I_EFF_LOGB_minus_sinusoidal|N=100|U4_effect_skill"]["ci95_background_bootstrap"][0]>0
                       for reg in ("IDLIKE","BACKGROUND_OOD"))
    active_gates={"I_eff_beats_sinusoidal_and_random_both_regimes":info_pass,"U4_skill_positive_both_regimes":u4_positive,"U4_skill_beats_sinusoidal_both_regimes":u4_beats_fixed}
    decision={"protocol":"Q1_ACTIVE_EXCITATION","fixed_policy_verdict":"unchanged; see Phase B","active_verdict":"PASS" if all(active_gates.values()) else "FAIL","active_gates":active_gates,
              "active_beats_sinusoidal_on_Ieff_logb":all(paired_comparisons[f"{reg}|ACTIVE_I_EFF_LOGB_minus_sinusoidal|N=100|I_eff_logb"]["ci95_background_bootstrap"][0]>0 for reg in ("IDLIKE","BACKGROUND_OOD")),
              "active_U4_skill_positive_both_regimes":u4_positive,
              "active_U4_beats_sinusoidal_both_regimes":u4_beats_fixed,
              "protected_test_used":False,"action_bound_violations":violations,"acquisition_candidate_evaluations":candidate_counts,"wall_seconds_by_policy":runtime_by_policy,
              "interpretation":"This separately named active protocol cannot turn the fixed-policy Q1 gate into a PASS."}
    (OUT/"summary.json").write_text(json.dumps({"metrics":summary,"paired_comparisons":paired_comparisons,"decision":decision},indent=2))
    (OUT/"DECISION.md").write_text("# Q1_ACTIVE_EXCITATION development decision\n\n"+json.dumps(decision,indent=2)+"\n\nActive results are separate from Q1_FIXED_POLICY. See the full per-background CSVs and `summary.json`.\n")
    (OUT/"provenance.json").write_text(json.dumps({"branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),"start_commit":"43245aee373ddd1061406a8c22f17f7ba686e30d","python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"device":"cpu","source_sha256":{str(Path(__file__).relative_to(ROOT)):sha(__file__),"q1_v5/active_design.py":sha(ROOT/"q1_v5/active_design.py")},"protected_test_used":False,"elapsed_seconds":time.time()-started},indent=2))
    print(json.dumps(decision,indent=2))


if __name__=="__main__":main()
