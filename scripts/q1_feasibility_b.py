"""Experiment B: probe whether frozen M2 old context recovers d and b."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q1.diagnostics.feasibility_abc import (
    fit_ridge_fixed, fit_ridge_probe, old_transition_statistics,
    predict_ridge_probe, vector_response,
)
from q1.simulator import apply_program
from q1.train_primitives import _get_m2
from scripts.q1_feasibility_common import (
    encode_old_context, get_stage_data, load_protocol, params_for, predict_m2,
    probe_rows, sha256_file, sha256_tree, write_json,
)


def _standard_effect(effect: np.ndarray) -> dict:
    energy = float(np.mean(effect ** 2))
    return {"effect_energy_delta_v_squared": energy,
            "effect_rms_delta_v": float(np.sqrt(energy)),
            "mean_component_effect": np.mean(effect, axis=0).tolist()}


def _ci(rows, key, z=1.96):
    bybg = {}
    for r in rows:
        bybg.setdefault(r["environment_id"], []).append(float(r[key]))
    vals = np.asarray([np.mean(v) for v in bybg.values()])
    mean = float(vals.mean())
    se = float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else 0.0
    return {"n_backgrounds": len(vals), "mean": mean, "ci95_background": [mean-z*se, mean+z*se]}


def _save_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run(config_path: str, out_path: str, device_str: str = "cpu", threads: int = 1) -> dict:
    cfg, manifest, out = load_protocol(config_path, out_path)
    for name in ("b_representation_metrics.json", "b_per_background.csv", "b_status.json"):
        if (out/name).exists(): raise FileExistsError(f"refusing to overwrite {out/name}")
    split = json.loads((out/"original_split_manifest.json").read_text(encoding="utf-8"))
    split_cfg = yaml.safe_load((ROOT/"configs/q1_splits.yaml").read_text(encoding="utf-8"))
    m2_path = ROOT/cfg["checkpoints"]["m2"]
    if not m2_path.is_file():
        write_json(out/"b_blocked.json", {"status":"BLOCKED_MISSING_M2_CHECKPOINT","missing":str(m2_path)})
        raise FileNotFoundError(f"required frozen M2 checkpoint is absent: {m2_path}")
    torch.set_num_threads(max(1,int(threads)))
    device=torch.device(device_str)
    model, normalizer, _ = _get_m2(m2_path,device)
    start=time.perf_counter()
    fit_idx=manifest["b_inner_train"]["indices"]
    val_idx=manifest["b_inner_validation"]["indices"]
    # Build B's separately declared 36/12 inner split from original train backgrounds.
    bfit=[]
    for idx in fit_idx:
        params=params_for(manifest,"TRAIN",idx,split,split_cfg["parameter_ranges"])
        seed=int(cfg["seed"])+idx*100003+300001
        old,_,_=get_stage_data(params,(),seed=seed,n_support=int(cfg["old_context_size"]),n_adapt=100,
            n_query=int(cfg["assessment_query_size"]),dt=float(cfg["dt"]),sigma=float(cfg["noise_sigma"]))
        context=encode_old_context(model,normalizer,old,device).detach().cpu().numpy()
        bfit.append({"index":idx,"context":context,"stats":old_transition_statistics(old.states[:-1],old.actions,old.delta_v),
                     "target":np.log([params.d,params.b])})
    bval=[]
    for idx in val_idx:
        params=params_for(manifest,"TRAIN",idx,split,split_cfg["parameter_ranges"])
        seed=int(cfg["seed"])+idx*100003+300001
        old,_,_=get_stage_data(params,(),seed=seed,n_support=int(cfg["old_context_size"]),n_adapt=100,
            n_query=int(cfg["assessment_query_size"]),dt=float(cfg["dt"]),sigma=float(cfg["noise_sigma"]))
        context=encode_old_context(model,normalizer,old,device).detach().cpu().numpy()
        bval.append({"index":idx,"context":context,"stats":old_transition_statistics(old.states[:-1],old.actions,old.delta_v),
                     "target":np.log([params.d,params.b])})
    models={}; selection={}
    for name,key in (("context_ridge","context"),("transition_statistics_ridge","stats")):
        candidate,meta=fit_ridge_probe(np.stack([x[key] for x in bfit]),np.stack([x["target"] for x in bfit]),
            np.stack([x[key] for x in bval]),np.stack([x["target"] for x in bval]),cfg["b_probe"]["ridge_grid"])
        ridge=meta["selected_ridge"]
        b_all=bfit+bval
        all_x=np.stack([x[key] for x in b_all]); all_y=np.stack([x["target"] for x in b_all])
        models[name]=fit_ridge_fixed(all_x,all_y,ridge)
        selection[name]=meta
    b_all=bfit+bval
    intercept_log=np.mean(np.stack([r["target"] for r in b_all]),axis=0)
    a_est={}
    for r in json.loads((out/"a_parameter_estimates.json").read_text(encoding="utf-8"))["backgrounds"]:
        if r["regime"] in ("IDLIKE_DEVELOPMENT","DEVELOPMENT_OOD"):
            a_est[(r["regime"],r["background"])]=np.asarray(r["theta_hat"],float)
    rows=[]
    eval_regimes=("IDLIKE_DEVELOPMENT","DEVELOPMENT_OOD")
    for ridx,regime in enumerate(eval_regimes):
        count=int(cfg["idlike_backgrounds"] if regime.startswith("IDLIKE") else cfg["development_ood_backgrounds"])
        for env in range(count):
            params=params_for(manifest,regime,env,split,split_cfg["parameter_ranges"])
            env_id=(manifest["idlike"]["environment_ids"][env] if regime.startswith("IDLIKE")
                    else manifest["development_ood"]["environment_ids"][env])
            seed=int(cfg["seed"])+ridx*4_000_000+env*100_003
            old,_,_=get_stage_data(params,(),seed=seed,n_support=int(cfg["old_context_size"]),n_adapt=100,
                n_query=int(cfg["assessment_query_size"]),dt=float(cfg["dt"]),sigma=float(cfg["noise_sigma"]))
            context=encode_old_context(model,normalizer,old,device).detach().cpu().numpy()
            stats=old_transition_statistics(old.states[:-1],old.actions,old.delta_v)
            preds={"intercept_only":np.exp(intercept_log),
                "context_ridge":np.exp(predict_ridge_probe(models["context_ridge"],context[None])[0]),
                "transition_statistics_ridge":np.exp(predict_ridge_probe(models["transition_statistics_ridge"],stats[None])[0]),
                "experiment_a_physics_estimate":a_est[(regime,env)][4:6]}
            for code in range(4):
                _,_,query=get_stage_data(params,(code,),seed=seed,n_support=int(cfg["old_context_size"]),
                    n_adapt=100,n_query=int(cfg["assessment_query_size"]),dt=float(cfg["dt"]),sigma=float(cfg["noise_sigma"]))
                probes=probe_rows(query)[:int(cfg["assessment_query_size"])]
                f0=vector_response(params,probes,dt=float(cfg["dt"]))
                f1=vector_response(apply_program(params,(code,)),probes,dt=float(cfg["dt"]))
                effect=f1-f0
                pred_old=predict_m2(model,normalizer,context,probes,device)
                old_mse=float(np.mean((pred_old-f0)**2))
                v=probes[:,2:4]; speed=np.linalg.norm(v,axis=1)
                design=np.c_[v,speed[:,None]*v]
                singular=np.linalg.svd(design,compute_uv=False)
                gram_cond=float((singular[0]/singular[-1])**2) if singular[-1]>0 else None
                for predictor,db in preds.items():
                    d_hat,b_hat=map(float,db)
                    # A fitted (d,b) plug-in with known other simulator parameters is not evaluated as a fair method;
                    # it is only parameter recoverability in this experiment.
                    row={"regime":regime,"environment_id":env_id,"background":env,
                         "physical_code":code,"program":[code],"predictor":predictor,
                         "d_true_reporting_only":params.d,"b_true_reporting_only":params.b,
                         "d_prediction":d_hat,"b_prediction":b_hat,
                         "log_db_mse":float(np.mean((np.log([d_hat,b_hat])-np.log([params.d,params.b]))**2)),
                         "old_law_conditional_mse_delta_v_squared":old_mse,
                         **_standard_effect(effect),
                         "effect_energy_to_old_model_error_ratio":float(np.mean(effect**2)/old_mse) if old_mse>0 else None,
                         "speed_p10":float(np.quantile(speed,.1)),"speed_median":float(np.median(speed)),
                         "speed_p90":float(np.quantile(speed,.9)),
                         "v_and_speedv_design_gram_condition":gram_cond,
                         "assessment_query_size":len(probes)}
                    rows.append(row)
    agg={}
    for regime in eval_regimes:
        for predictor in ("intercept_only","context_ridge","transition_statistics_ridge","experiment_a_physics_estimate"):
            for metric in ("log_db_mse","old_law_conditional_mse_delta_v_squared","effect_energy_to_old_model_error_ratio"):
                sub=[r for r in rows if r["regime"]==regime and r["predictor"]==predictor]
                agg[f"{regime}/{predictor}/{metric}"]=_ci(sub,metric)
    metric_file={"status":"DEVELOPMENT_DIAGNOSTIC","m2_checkpoint_sha256":sha256_file(m2_path),
        "ridge_selection_train_only":selection,"normalization":"targets log(d),log(b); standardized within training backgrounds",
        "metrics":agg,"per_code_old_error_effect_scale":{},
        "limits":["linear probe failure does not establish absence of nonlinear context information",
                  "positive probe does not prove a global latent transformation", "validation regime is DEVELOPMENT_OOD"]}
    for regime in eval_regimes:
        for code in (2,3):
            sub=[r for r in rows if r["regime"]==regime and r["physical_code"]==code and r["predictor"]=="context_ridge"]
            metric_file["per_code_old_error_effect_scale"][f"{regime}/code{code}"]={
                "mean_old_law_mse":float(np.mean([r["old_law_conditional_mse_delta_v_squared"] for r in sub])),
                "mean_effect_energy":float(np.mean([r["effect_energy_delta_v_squared"] for r in sub])),
                "median_speed":float(np.median([r["speed_median"] for r in sub])),
                "median_v_speedv_gram_condition":float(np.median([r["v_and_speedv_design_gram_condition"] for r in sub]))}
    protected=[ROOT/"results/q1/u4_root_cause_v1",ROOT/"results/q1/repair_v1_m3_effect_ablation"]
    # Only new output has been written; verify untouched historical trees against A's snapshot.
    a_status=json.loads((out/"a_status.json").read_text(encoding="utf-8"))
    current={str(p.relative_to(ROOT)):sha256_tree(p) for p in protected if p.exists()}
    if a_status["protected_tree_hashes_before"]!=current: raise RuntimeError("protected legacy artifacts changed")
    write_json(out/"b_representation_metrics.json",metric_file)
    _save_csv(out/"b_per_background.csv",rows)
    write_json(out/"b_status.json",{"status":"COMPLETE","wall_seconds":time.perf_counter()-start,
        "evaluation_rows":len(rows),"training_contexts":len(b_all),"checkpoint_sha256":sha256_file(m2_path)})
    return {"rows":len(rows),"wall_seconds":time.perf_counter()-start}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/q1_feasibility_abc.yaml")
    ap.add_argument("--out",default="results/q1/feasibility_abc_v1"); ap.add_argument("--device",default="cpu")
    ap.add_argument("--threads",type=int,default=1); args=ap.parse_args()
    result=run(args.config,args.out,args.device,args.threads); print(json.dumps(result))


if __name__=="__main__": main()
