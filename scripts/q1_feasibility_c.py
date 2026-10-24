"""Experiment C: bounded train-only direct function-space effect diagnostics."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q1.datasets import rollout
from q1.diagnostics.feasibility_abc import (
    effect_metrics, estimate_old_params, params_from_vector, vector_response,
)
from q1.direct_effect import DirectEffectMLP, inference_effect
from q1.model import TransitionNormalizer
from q1.primitives import PrimitiveBank
from q1.simulator import apply_program
from q1.train_primitives import _get_m2
from scripts.q1_feasibility_common import (
    encode_old_context, get_stage_data, load_protocol, params_for, predict_m2,
    probe_rows, sha256_file, sha256_tree, write_json,
)


def make_effect_training_target(params, physical_code: int, probes: np.ndarray, *, dt: float) -> np.ndarray:
    """Privileged training-target generation only; never called by deployed inference."""
    from q1.simulator import response
    x=np.asarray(probes,dtype=np.float64)
    return response(apply_program(params,(int(physical_code),)),x,dt=dt)-response(params,x,dt=dt)


def _as_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _batch_inputs(rows, code_map, context_mean, context_scale, normalizer, device):
    xs=[]; cs=[]; codes=[]; groups=[]; ys=[]
    for row in rows:
        p=torch.as_tensor(row["probes"],dtype=torch.float32,device=device)
        xn=normalizer.normalize_x(p)
        c=(row["context"]-context_mean)/context_scale
        for group in range(4):
            n=len(p)
            xs.append(xn)
            cs.append(c.reshape(1,-1).expand(n,-1))
            codes.append(torch.full((n,),int(code_map[str(group)]),dtype=torch.long,device=device))
            groups.append(torch.full((n,),group,dtype=torch.long,device=device))
            ys.append(torch.as_tensor(row["targets"][group],dtype=torch.float32,device=device))
    return {"x":torch.cat(xs),"c":torch.cat(cs),"code":torch.cat(codes),
            "group":torch.cat(groups),"y":torch.cat(ys)}


def _make_dataset(indices, manifest, split, split_cfg, cfg, model, normalizer, device,
                  *, seed_offset: int, training: bool):
    rows=[]
    for idx in indices:
        params=params_for(manifest,"TRAIN",int(idx),split,split_cfg["parameter_ranges"])
        seed=int(cfg["seed"])+int(idx)*100003+int(seed_offset)
        old,_,_=get_stage_data(params,(),seed=seed,n_support=int(cfg["old_context_size"]),
            n_adapt=int(cfg["adaptation_trajectory_size"]),n_query=int(cfg["c_direct_effect"]["train_probes_per_code_background"]),
            dt=float(cfg["dt"]),sigma=float(cfg["noise_sigma"]))
        context=encode_old_context(model,normalizer,old,device).detach().cpu()
        for physical_code in range(4):
            _,_,query=get_stage_data(params,(physical_code,),seed=seed,n_support=int(cfg["old_context_size"]),
                n_adapt=int(cfg["adaptation_trajectory_size"]),
                n_query=int(cfg["c_direct_effect"]["train_probes_per_code_background"]),
                dt=float(cfg["dt"]),sigma=float(cfg["noise_sigma"]))
            probes=probe_rows(query)[:int(cfg["c_direct_effect"]["train_probes_per_code_background"])]
            target=make_effect_training_target(params,physical_code,probes,dt=float(cfg["dt"]))
            rows.append({"index":int(idx),"environment_id":split["environment_ids"]["train"][int(idx)],
                "physical_code":physical_code,"context":context,"probes":probes,"targets":{physical_code:target}})
    return rows


def _model_arrays(dataset, code_map, context_mean, context_scale, normalizer, scales_by_opaque, device):
    xs=[];cs=[];codes=[];groups=[];ys=[]
    for row in dataset:
        p=torch.as_tensor(row["probes"],dtype=torch.float32,device=device)
        xn=normalizer.normalize_x(p)
        ctx=((row["context"].to(device)-context_mean)/context_scale).reshape(1,-1).expand(len(p),-1)
        physical=int(row["physical_code"])
        code=int(code_map[str(physical)])
        target=torch.as_tensor(row["targets"][physical],dtype=torch.float32,device=device)
        xs.append(xn); cs.append(ctx); codes.append(torch.full((len(p),),code,dtype=torch.long,device=device))
        groups.append(torch.full((len(p),),physical,dtype=torch.long,device=device)); ys.append(target/scales_by_opaque[code])
    return {"x":torch.cat(xs),"c":torch.cat(cs),"code":torch.cat(codes),
            "group":torch.cat(groups),"y":torch.cat(ys)}


def _train_one(train_ds, val_ds, *, model_cfg, code_map, context_mean, context_scale,
               normalizer, scales_phys, seed, variant, cfg, device, checkpoint_path):
    torch.manual_seed(int(seed))
    context_dim=int(context_mean.numel())
    net=DirectEffectMLP(context_dim,hidden_dim=int(model_cfg["hidden_dim"]),
        embedding_dim=int(model_cfg["code_embedding_dim"]),use_context=(variant=="context_aware")).to(device)
    # map physical-code scales to checkpoint opaque ids for model output calibration
    scales_opaque=torch.empty(4,dtype=torch.float32,device=device)
    for group in range(4): scales_opaque[int(code_map[str(group)])]=float(scales_phys[group])
    train=_model_arrays(train_ds,code_map,context_mean,context_scale,normalizer,scales_opaque,device)
    val=_model_arrays(val_ds,code_map,context_mean,context_scale,normalizer,scales_opaque,device)
    indices={g:torch.where(train["group"]==g)[0].cpu().numpy() for g in range(4)}
    rng=np.random.default_rng(int(seed)+992)
    steps=int(model_cfg["max_optimizer_steps"]); batch=int(model_cfg["batch_per_code"])
    schedules={g:rng.choice(indices[g],size=(steps,batch),replace=True) for g in range(4)}
    optimizer=torch.optim.Adam(net.parameters(),lr=float(model_cfg["learning_rate"]))
    best_loss=float("inf"); best_state=None; best_step=0; curves=[]; gradient_rows=[]; finite=True
    for step in range(1,steps+1):
        chosen=np.concatenate([schedules[g][step-1] for g in range(4)])
        ix=torch.as_tensor(chosen,dtype=torch.long,device=device)
        xb,cb,zb,yb=(train[k][ix] for k in ("x","c","code","y"))
        net.train(); pred=net(xb,cb,zb)
        code_losses=[]
        for g in range(4):
            mask=train["group"][ix]==g
            code_losses.append(torch.mean((pred[mask]-yb[mask])**2))
        loss=torch.stack(code_losses).mean()
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite training loss at {variant} seed {seed} step {step}")
        if step % int(model_cfg["gradient_diagnostics_every"])==0 or step==1:
            norms={}
            params=tuple(p for p in net.parameters() if p.requires_grad)
            for g,closs in enumerate(code_losses):
                grads=torch.autograd.grad(closs,params,retain_graph=True,allow_unused=True)
                sq=sum(float(torch.sum(gv.detach()**2)) for gv in grads if gv is not None)
                norms[str(g)]=math.sqrt(sq)
                if not math.isfinite(norms[str(g)]): finite=False
            gradient_rows.append({"step":step,"per_physical_code_parameter_grad_norm":norms,"finite":finite})
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise FloatingPointError(f"nonfinite parameter gradient at {variant} seed {seed} step {step}")
        optimizer.step()
        every=int(model_cfg["validation_every"])
        if step%every==0 or step==steps:
            net.eval()
            with torch.no_grad():
                vp=net(val["x"],val["c"],val["code"])
                vlosses={str(g):float(torch.mean((vp[val["group"]==g]-val["y"][val["group"]==g])**2)) for g in range(4)}
                tlosses={str(g):float(code_losses[g].detach()) for g in range(4)}
                val_mean=float(np.mean(list(vlosses.values())))
            curves.append({"step":step,"train_normalized_mse_by_physical_code":tlosses,
                           "inner_validation_normalized_mse_by_physical_code":vlosses,
                           "inner_validation_mean_normalized_mse":val_mean})
            if val_mean<best_loss:
                best_loss=val_mean;best_step=step;best_state=copy.deepcopy(net.state_dict())
    if best_state is None: raise RuntimeError("no finite inner-validation model state was selected")
    net.load_state_dict(best_state); net.eval()
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":best_state,"seed":int(seed),"variant":variant,"best_step":best_step,
        "inner_validation_loss":best_loss,"context_mean":_as_np(context_mean),
        "context_scale":_as_np(context_scale),"effect_scales_physical_order":np.asarray(scales_phys),
        "group_to_code":code_map,"model_config":model_cfg,"training_label_scope":"privileged paired simulator conditional means"},checkpoint_path)
    return net,{"seed":int(seed),"variant":variant,"best_step":best_step,
        "best_inner_validation_normalized_mse":best_loss,"training_curve":curves,
        "gradient_diagnostics":gradient_rows,"finite_gradient_checks_passed":finite,
        "checkpoint":checkpoint_path.relative_to(ROOT).as_posix(),"checkpoint_sha256":sha256_file(checkpoint_path)}


def _evaluate(model, context_aware_net, context_ablation_net, *, normalizer, bank, m3_map, Gmap, scales_opaque,
              context_mean,context_scale,params,theta_hat,physical_code,probes,dt,device,seed,regime,env,env_id):
    from q1.diagnostics.feasibility_abc import effect_metrics
    code_id=int(m3_map[str(physical_code)])
    x=torch.as_tensor(probes,dtype=torch.float32,device=device)
    xn=normalizer.normalize_x(x)
    c0=Gmap.reshape(1,-1).expand(len(x),-1)
    cn=(c0-context_mean)/context_scale
    pred_old=predict_m2(model,normalizer,Gmap,probes,device)
    true_old=vector_response(params,probes,dt=dt)
    from q1.simulator import apply_program
    after=apply_program(params,(physical_code,))
    true_new=vector_response(after,probes,dt=dt)
    effect_true=true_new-true_old
    with torch.no_grad():
        c0_t=Gmap.reshape(1,-1)
        transformed=bank.apply_one(c0_t,code_id)
        pred_m3=normalizer.denormalize_dv(model.executor(xn,transformed.expand(len(x),-1))).cpu().numpy()
        zero=np.zeros_like(effect_true)
        effects={"zero_effect":zero,"m3_beta025":pred_m3-pred_old}
        for name,net in (("direct_effect_context",context_aware_net),
                         ("direct_effect_context_ablated",context_ablation_net)):
            codes=torch.full((len(x),),code_id,dtype=torch.long,device=device)
            normalized=net(xn,cn,codes)
            effect=normalized*scales_opaque[codes].reshape(-1,1)
            effects[name]=effect.cpu().numpy().astype(np.float64)
    theta_after=apply_program(theta_hat,(physical_code,))
    effects["a2_physics_plugin_privileged"]=vector_response(theta_after,probes,dt=dt)-vector_response(theta_hat,probes,dt=dt)
    result=[]
    for method,eff in effects.items():
        m=effect_metrics(true_old,true_new,pred_old,eff)
        bias=np.mean(eff-effect_true,axis=0)
        denom=float(np.linalg.norm(effect_true))
        calibration=float(np.linalg.norm(eff)/denom) if denom>1e-15 else None
        result.append({"regime":regime,"environment_id":env_id,"background":env,
            "physical_code":physical_code,"method":method,"seed":seed,**m,
            "effect_bias_component_mean_delta_v":bias.tolist(),"effect_bias_norm_delta_v":float(np.linalg.norm(bias)),
            "effect_calibration_norm_ratio":calibration,
            "effect_sign_agreement":float(np.mean(np.sign(eff)==np.sign(effect_true))),
            "privileged_physics_model_baseline":method=="a2_physics_plugin_privileged"})
    return result


def _aggregate(rows):
    out={}
    for regime in ("IDLIKE_DEVELOPMENT","DEVELOPMENT_OOD"):
        for code in range(4):
            for method in sorted({r["method"] for r in rows}):
                subset=[r for r in rows if r["regime"]==regime and r["physical_code"]==code and r["method"]==method]
                bybg={}
                for r in subset:bybg.setdefault(r["environment_id"],[]).append(r)
                bgrows=[]
                for env,group in bybg.items():
                    bgrows.append({k:float(np.mean([x[k] for x in group if x[k] is not None]))
                        for k in ("effect_skill","effect_mse_delta_v_squared","effect_energy_delta_v_squared",
                                  "after_error_delta_v_squared","learned_after_gain","perfect_effect_gain")})
                skills=np.asarray([x["effect_skill"] for x in bgrows if np.isfinite(x["effect_skill"])])
                mean=float(skills.mean()) if len(skills) else None
                se=float(skills.std(ddof=1)/np.sqrt(len(skills))) if len(skills)>1 else 0.0
                out[f"{regime}/code{code}/{method}"]={"n_backgrounds":len(bgrows),
                    "mean_background_effect_skill":mean,
                    "effect_skill_ci95_background":[mean-1.96*se,mean+1.96*se] if mean is not None else None,
                    "mean_background_effect_mse":float(np.mean([r["effect_mse_delta_v_squared"] for r in bgrows])) if bgrows else None,
                    "mean_background_effect_energy":float(np.mean([r["effect_energy_delta_v_squared"] for r in bgrows])) if bgrows else None,
                    "ratio_of_mean_effect_error_to_energy_skill":float(1-np.mean([r["effect_mse_delta_v_squared"] for r in bgrows])/
                        np.mean([r["effect_energy_delta_v_squared"] for r in bgrows])) if bgrows and np.mean([r["effect_energy_delta_v_squared"] for r in bgrows])>0 else None,
                    "mean_after_mse_gain":float(np.mean([r["learned_after_gain"] for r in bgrows])) if bgrows else None,
                    "worst_background_effect_skill":min((r["effect_skill"] for r in bgrows),default=None),
                    "background_signs_positive":sum(r["effect_skill"]>0 for r in bgrows),
                    "uncertainty_unit":"background; model seeds averaged within background"}
    return out


def run(config_path,out_path,device_str="cpu",threads=1):
    cfg,manifest,out=load_protocol(config_path,out_path)
    for name in ("c_direct_effect_metrics.json","c_per_background.csv","c_training_curves.json","c_gradient_diagnostics.json","c_status.json"):
        if (out/name).exists():raise FileExistsError(f"refusing to overwrite {out/name}")
    split=json.loads((out/"original_split_manifest.json").read_text(encoding="utf-8"))
    split_cfg=yaml.safe_load((ROOT/"configs/q1_splits.yaml").read_text(encoding="utf-8"))
    m2_path=ROOT/cfg["checkpoints"]["m2"];m3_path=ROOT/cfg["checkpoints"]["m3"]
    missing=[str(p) for p in (m2_path,m3_path) if not p.is_file()]
    if missing:
        write_json(out/"c_blocked.json",{"status":"BLOCKED_MISSING_REQUIRED_CHECKPOINT","missing":missing})
        raise FileNotFoundError("required local checkpoint absent; no random or downloaded fallback: "+", ".join(missing))
    torch.set_num_threads(max(1,int(threads)));device=torch.device(device_str)
    model,normalizer,m2_data=_get_m2(m2_path,device)
    m3_data=torch.load(m3_path,map_location=device,weights_only=False)
    bank=PrimitiveBank(int(m2_data["config"]["model"]["context_dim"]),hidden_dim=64).to(device)
    bank.load_state_dict(m3_data["primitives"]);bank.eval()
    for p in bank.parameters():p.requires_grad_(False)
    code_map=m3_data["group_to_code"]
    if set(code_map)!={str(i) for i in range(4)} or sorted(map(int,code_map.values()))!=list(range(4)):
        raise ValueError("M3 checkpoint group_to_code is not a bijection")
    m2_params=[p for p in model.parameters()]
    if any(p.requires_grad for p in m2_params):raise RuntimeError("M2 must remain frozen during C")
    started=time.perf_counter()
    train_idx=manifest["c_training_fit"]["indices"];val_idx=manifest["c_training_inner_validation"]["indices"]
    training=_make_dataset(train_idx,manifest,split,split_cfg,cfg,model,normalizer,device,
        seed_offset=450001,training=True)
    inner=_make_dataset(val_idx,manifest,split,split_cfg,cfg,model,normalizer,device,
        seed_offset=450001,training=True)
    contexts=torch.stack([x["context"] for x in training[::4]])
    context_mean=contexts.mean(0).to(device)
    context_scale=contexts.std(0,unbiased=False).clamp_min(1e-5).to(device)
    # Only fit-partition target energies define physical per-code scales.
    scales_phys=np.asarray([np.sqrt(np.mean(np.concatenate([r["targets"][g] for r in training if r["physical_code"]==g])**2))
                            for g in range(4)],dtype=np.float64)
    if not np.isfinite(scales_phys).all() or (scales_phys<=0).any():raise ValueError("invalid train-only per-code effect scales")
    scales_opaque=np.empty(4,dtype=np.float32)
    for g in range(4):scales_opaque[int(code_map[str(g)])]=scales_phys[g]
    model_cfg={**cfg["c_direct_effect"],"max_optimizer_steps":int(cfg["c_direct_effect"]["max_optimizer_steps"])}
    checkpoints=out/"checkpoints";runs=[];curves=[];grads=[];models={}
    for seed in cfg["c_direct_effect"]["optimizer_seeds"]:
        for variant in ("context_aware","context_ablated"):
            stem=f"direct_effect_{variant}_seed{seed}"
            ckpt=checkpoints/f"{stem}.pt"
            attempt=2
            while ckpt.exists():
                ckpt=checkpoints/f"{stem}_retry{attempt}.pt"
                attempt+=1
            net,meta=_train_one(training,inner,model_cfg=model_cfg,code_map=code_map,
                context_mean=context_mean,context_scale=context_scale,normalizer=normalizer,
                scales_phys=scales_phys,seed=int(seed),variant=variant,cfg=cfg,device=device,checkpoint_path=ckpt)
            runs.append(meta);models[(int(seed),variant)]=net
            curves.append({"seed":int(seed),"variant":variant,"curve":meta["training_curve"]})
            grads.append({"seed":int(seed),"variant":variant,"diagnostics":meta["gradient_diagnostics"]})
    scales_t=torch.as_tensor(scales_opaque,dtype=torch.float32,device=device)
    a_est={}
    for r in json.loads((out/"a_parameter_estimates.json").read_text(encoding="utf-8"))["backgrounds"]:
        a_est[(r["regime"],r["background"])]=params_from_vector(r["theta_hat"])
    rows=[]
    for ridx,regime in enumerate(("IDLIKE_DEVELOPMENT","DEVELOPMENT_OOD")):
        count=int(cfg["idlike_backgrounds"] if regime.startswith("IDLIKE") else cfg["development_ood_backgrounds"])
        for env in range(count):
            params=params_for(manifest,regime,env,split,split_cfg["parameter_ranges"])
            env_id=(manifest["idlike"]["environment_ids"][env] if regime.startswith("IDLIKE") else manifest["development_ood"]["environment_ids"][env])
            seed=int(cfg["seed"])+ridx*4_000_000+env*100_003
            old=rollout(params,n=int(cfg["old_context_size"]),seed=seed,dt=float(cfg["dt"]),sigma=float(cfg["noise_sigma"]))
            context=encode_old_context(model,normalizer,old,device)
            for code in range(4):
                _,_,query=get_stage_data(params,(code,),seed=seed,n_support=int(cfg["old_context_size"]),
                    n_adapt=int(cfg["adaptation_trajectory_size"]),n_query=int(cfg["assessment_query_size"]),
                    dt=float(cfg["dt"]),sigma=float(cfg["noise_sigma"]))
                probes=probe_rows(query)[:int(cfg["assessment_query_size"])]
                true_old=vector_response(params,probes,dt=float(cfg["dt"]))
                for cseed in cfg["c_direct_effect"]["optimizer_seeds"]:
                    res=_evaluate(model,models[(int(cseed),"context_aware")],models[(int(cseed),"context_ablated")],
                        normalizer=normalizer,bank=bank,m3_map=code_map,Gmap=context,
                        scales_opaque=scales_t,context_mean=context_mean,context_scale=context_scale,
                        params=params,theta_hat=a_est[(regime,env)],physical_code=code,
                        probes=probes,dt=float(cfg["dt"]),device=device,seed=int(cseed),
                        regime=regime,env=env,env_id=env_id)
                    rows.extend(res)
    agg=_aggregate(rows)
    legacy=[ROOT/"results/q1/u4_root_cause_v1",ROOT/"results/q1/repair_v1_m3_effect_ablation"]
    a_status=json.loads((out/"a_status.json").read_text(encoding="utf-8"))
    current={str(p.relative_to(ROOT)):sha256_tree(p) for p in legacy if p.exists()}
    if a_status["protected_tree_hashes_before"]!=current:raise RuntimeError("protected legacy artifacts changed during C")
    metric={"status":"DEVELOPMENT_DIAGNOSTIC","method":"direct single-effect prediction; composition not evaluated",
        "checkpoint_hashes":{"m2":sha256_file(m2_path),"m3":sha256_file(m3_path)},
        "group_to_code":code_map,"training_supervision":"privileged paired simulator conditional-mean effects on training backgrounds only",
        "inference_inputs":"normalized state-action, frozen old M2 context, opaque code; no true parameters/program labels/after trajectory",
        "train_only_effect_scales_physical_order":scales_phys.tolist(),
        "selection":"best mean normalized effect MSE on 8 inner validation backgrounds from original train split",
        "models":runs,"aggregates":agg,"test_or_ood_split_used":False,
        "development_ood_is_previously_inspected":True}
    write_json(out/"c_direct_effect_metrics.json",metric)
    with (out/"c_per_background.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(out/"c_training_curves.json",curves);write_json(out/"c_gradient_diagnostics.json",grads)
    write_json(out/"c_status.json",{"status":"COMPLETE","wall_seconds":time.perf_counter()-started,
        "training_seeds":cfg["c_direct_effect"]["optimizer_seeds"],"checkpoint_hashes":{x["checkpoint"]:x["checkpoint_sha256"] for x in runs},
        "protected_tree_hashes_before":a_status["protected_tree_hashes_before"],"protected_tree_hashes_after":current})
    return {"rows":len(rows),"models":len(runs),"wall_seconds":time.perf_counter()-started}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/q1_feasibility_abc.yaml")
    ap.add_argument("--out",default="results/q1/feasibility_abc_v1");ap.add_argument("--device",default="cpu")
    ap.add_argument("--threads",type=int,default=1);args=ap.parse_args()
    print(json.dumps(run(args.config,args.out,args.device,args.threads)))


if __name__=="__main__":main()
