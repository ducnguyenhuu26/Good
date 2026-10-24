"""Privileged, offline function-space projections for Q1 V2 root-cause diagnosis.

This diagnostic may query simulator conditional means. It is not a learned-model
training path and never reads original test environments.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from q1.datasets import rollout
from q1.simulator import DynamicsParams,apply_program,response
from q1_v2.basis import PartitionedBasis
from q1_v2.training import design
from scripts.q1_v2_pipeline import collect_backgrounds,gather_forbidden_seeds

RIDGE_GRID=(0.0,1e-8,1e-6,1e-4,1e-2)
DT=.05;SIGMA=.0005

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def x_from(traj):return np.c_[traj.states[:-1],traj.actions].astype(np.float32)

def fit_lstsq(phi,y,ridge=0.0):
    # Phi rows follow flatten([transition, output-coordinate, coefficient]); y has
    # the matching transition-major output order, in simulator delta-v units.
    phi=torch.as_tensor(phi,dtype=torch.float64);y=torch.as_tensor(y,dtype=torch.float64).reshape(-1)
    gram=phi.T@phi;scale=float(torch.trace(gram)/gram.shape[0])
    mat=gram+(float(ridge)*max(scale,1e-12))*torch.eye(gram.shape[0],dtype=phi.dtype)
    rhs=phi.T@y
    w=torch.linalg.solve(mat,rhs)
    sv=torch.linalg.svdvals(phi)
    rank=int(torch.linalg.matrix_rank(phi));cond=float(sv.max()/sv.min().clamp_min(1e-30))
    return w,rank,cond

def predict(phi,w):
    return (phi@w).reshape(-1,2)

def skill(pred,true):
    signal=float(np.mean(true**2));mse=float(np.mean((pred-true)**2))
    return (1.-mse/signal if signal>1e-24 else None),mse,signal

def choose_ridge(phi_fit,y_fit,phi_inner,y_inner,*,joint=False):
    phi_fit=torch.as_tensor(phi_fit,dtype=torch.float64)
    phi_inner=torch.as_tensor(phi_inner,dtype=torch.float64)
    scored=[]
    k=phi_fit.shape[1]
    if not joint:
        y_fit=torch.as_tensor(y_fit,dtype=torch.float64)
        y_inner=torch.as_tensor(y_inner,dtype=torch.float64)
        for ridge in RIDGE_GRID:
            w,rank,cond=fit_lstsq(phi_fit,y_fit,ridge)
            mse=float(torch.mean((phi_inner@w-y_inner).square()))
            scored.append((mse,ridge,w,rank,cond))
    else:
        y_fit=tuple(torch.as_tensor(y,dtype=torch.float64) for y in y_fit)
        y_inner=tuple(torch.as_tensor(y,dtype=torch.float64) for y in y_inner)
        # Shared nuisance baseline w0 plus effect displacement Delta w. Old law
        # receives w0; after law receives w0+Delta w. Penalize only Delta w.
        zeros=torch.zeros_like(phi_fit)
        design_fit=torch.cat((torch.cat((phi_fit,zeros),1),
                              torch.cat((phi_fit,phi_fit),1)),0)
        target_fit=torch.cat((y_fit[0],y_fit[1]),0)
        zeros_inner=torch.zeros_like(phi_inner)
        design_inner=torch.cat((torch.cat((phi_inner,zeros_inner),1),
                                torch.cat((phi_inner,phi_inner),1)),0)
        target_inner=torch.cat((y_inner[0],y_inner[1]),0)
        singular=torch.linalg.svdvals(design_fit)
        rank=int(torch.linalg.matrix_rank(design_fit))
        cond=float(singular.max()/singular.min().clamp_min(1e-30))
        for ridge in RIDGE_GRID:
            # Joint ridge is applied only to the effect block using the same
            # trace-scaled penalty convention as B.
            gram=design_fit.T@design_fit
            scale=float(torch.trace(phi_fit.T@phi_fit)/k)
            penalty=torch.zeros((2*k,2*k),dtype=gram.dtype);penalty[k:,k:]=torch.eye(k,dtype=gram.dtype)
            sol=torch.linalg.solve(gram+float(ridge)*max(scale,1e-12)*penalty,
                                   design_fit.T@target_fit)
            pred=design_inner@sol
            mse=float(torch.mean((pred-target_inner).square()))
            scored.append((mse,ridge,sol,rank,cond))
    scored.sort(key=lambda z:(z[0],z[1]))
    return scored[0],scored

def bootstrap_ci(values,seed=99117,n=5000):
    arr=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    sims=arr[rng.integers(0,len(arr),size=(n,len(arr)))].mean(axis=1)
    return [float(np.quantile(sims,.025)),float(np.quantile(sims,.975))]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-run',default='results/q1/q1_v2_operator_retrain_dev1')
    ap.add_argument('--output-dir',default='results/q1/q1_v2_root_repair_dev3/diagnostics/direct_effect_projection')
    ap.add_argument('--seed',type=int,default=20261030)
    ap.add_argument('--backgrounds-per-regime',type=int,default=8)
    ap.add_argument('--query-probes',type=int,default=1024)
    args=ap.parse_args()
    if args.backgrounds_per_regime<4:raise ValueError('use at least four backgrounds per regime')
    if args.query_probes<512:raise ValueError('query probes must be independent and >=512')
    src=ROOT/args.source_run;out=ROOT/args.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'refusing to overwrite diagnostic outputs: {out}')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    cfg=yaml.safe_load((src/'config.resolved.yaml').read_text(encoding='utf-8'))
    basis_ckpt=src/'basis_pretrained.pt';saved=torch.load(basis_ckpt,map_location='cpu',weights_only=False)
    basis=PartitionedBasis(hidden=int(cfg['hidden'])).cpu();basis.load_state_dict(saved['basis']);basis.eval()
    for p in basis.parameters():p.requires_grad_(False)
    forbidden,sources=gather_forbidden_seeds()
    id_seeds,id_params=collect_backgrounds(args.backgrounds_per_regime,args.seed,forbidden,'train',
        {'train':((.75,1.),(.28,.42),(.08,.14),(.18,.28))})
    forbidden.update(id_seeds)
    ood_ranges={'diagnostic_ood':{'B_diag':(1.08,1.2),'B_offdiag':(-.12,.12),'d':(.49,.59),'b':(.16,.20),'kappa':(.32,.39)}}
    ood_seeds,ood_params=collect_backgrounds(args.backgrounds_per_regime,args.seed+1,forbidden,'diagnostic_ood',ood_ranges)
    (out/'protocol.json').write_text(json.dumps({'protocol':'new diagnostic-only backgrounds; no test splits',
      'branch_sha_before_run':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'seed':args.seed,'backgrounds_per_regime':args.backgrounds_per_regime,'fit_budgets':[128,512,1024],
      'query_probes':args.query_probes,'dt':DT,'sigma':SIGMA,'U3_U4_only':True,'ridge_grid':RIDGE_GRID,
      'conditional_mean_targets_allowed_only_in_this_offline_diagnostic':True,'test_split_used':False,
      'collision_audit_sources':sources,'IDLIKE':{'seeds':id_seeds,'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in id_params]},
      'BACKGROUND_OOD':{'seeds':ood_seeds,'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in ood_params]}},indent=2),encoding='utf-8')
    # Train-split mean drag is used only for the explicitly privileged analytic U4 control.
    train_bs=[float(x['b']) for x in json.loads((src/'split_manifest.json').read_text(encoding='utf-8'))['train']['backgrounds']]
    train_b_mean=float(np.mean(train_bs))
    records=[];curves=[]
    for regime,seed_list,params_list in [('IDLIKE',id_seeds,id_params),('BACKGROUND_OOD',ood_seeds,ood_params)]:
      for bi,(bg_seed,before) in enumerate(zip(seed_list,params_list)):
        bg_id=f'{regime}-{bi:03d}'
        for code in (2,3):
          after=apply_program(before,(code,))
          # Existing A: independent, noisy pre/post trajectories and separate law fits.
          old_obs=rollout(before,n=1024,seed=bg_seed+2100000+code*101,dt=DT,sigma=SIGMA)
          new_obs=rollout(after,n=1024,seed=bg_seed+2200000+code*101,dt=DT,sigma=SIGMA)
          old_x,new_x=x_from(old_obs),x_from(new_obs)
          with torch.no_grad():
            phi_old=design(basis,torch.as_tensor(old_x)).numpy().reshape(-1,12)
            phi_new=design(basis,torch.as_tensor(new_x)).numpy().reshape(-1,12)
          # Independent inner and query trajectories: noisy states/actions, but no
          # query conditional means enter ridge selection or any fitted coefficient.
          fit_old=rollout(before,n=1024,seed=bg_seed+2300000+code*101,dt=DT,sigma=SIGMA)
          fit_new=rollout(after,n=1024,seed=bg_seed+2400000+code*101,dt=DT,sigma=SIGMA)
          inner_old=rollout(before,n=256,seed=bg_seed+2500000+code*101,dt=DT,sigma=SIGMA)
          inner_new=rollout(after,n=256,seed=bg_seed+2600000+code*101,dt=DT,sigma=SIGMA)
          q_old=rollout(before,n=args.query_probes//2,seed=bg_seed+2700000+code*101,dt=DT,sigma=SIGMA)
          q_new=rollout(after,n=args.query_probes-args.query_probes//2,seed=bg_seed+2800000+code*101,dt=DT,sigma=SIGMA)
          xfit=np.r_[x_from(fit_old)[:512],x_from(fit_new)[:512]]
          xinner=np.r_[x_from(inner_old),x_from(inner_new)]
          xquery=np.r_[x_from(q_old),x_from(q_new)]
          yold_fit=response(before,xfit,dt=DT).astype(np.float32).reshape(-1)
          ynew_fit=response(after,xfit,dt=DT).astype(np.float32).reshape(-1)
          yold_inner=response(before,xinner,dt=DT).astype(np.float32).reshape(-1)
          ynew_inner=response(after,xinner,dt=DT).astype(np.float32).reshape(-1)
          true_query=(response(after,xquery,dt=DT)-response(before,xquery,dt=DT)).astype(np.float64)
          with torch.no_grad():
            phifit=design(basis,torch.as_tensor(xfit)).numpy().reshape(-1,12)
            phiinner=design(basis,torch.as_tensor(xinner)).numpy().reshape(-1,12)
            phiq=design(basis,torch.as_tensor(xquery)).numpy().reshape(-1,12)
          # A separate-law OLS under noisy observed transition labels at the budget.
          for budget in (128,512,1024):
            wa,rank_a,cond_a=fit_lstsq(phi_old[:2*budget],old_obs.delta_v[:budget].reshape(-1))
            wb,rank_a_after,cond_a_after=fit_lstsq(phi_new[:2*budget],new_obs.delta_v[:budget].reshape(-1))
            pred_a=predict(phiq,(wb-wa).numpy())
            score_a,mse_a,signal=skill(pred_a,true_query)
            # B: direct paired conditional-mean effect projection; choose ridge on
            # inner diagnostics only, then lock it before querying fresh probes.
            del_fit=(ynew_fit-yold_fit)
            best_b,candidates_b=choose_ridge(phifit[:2*budget],del_fit[:2*budget],phiinner,
                torch.as_tensor(ynew_inner-yold_inner,dtype=torch.float64))
            _,ridge_b,delta_b,rank_b,cond_b=best_b
            pred_b=predict(phiq,delta_b.numpy());score_b,mse_b,_=skill(pred_b,true_query)
            # C: joint nuisance-canceling fit of common baseline and displacement.
            # The selected ridge is validated only on inner conditional-mean probes.
            joint_fit=(yold_fit[:2*budget],ynew_fit[:2*budget]);joint_inner=(torch.as_tensor(yold_inner,dtype=torch.float64),torch.as_tensor(ynew_inner,dtype=torch.float64))
            best_c,candidates_c=choose_ridge(phifit[:2*budget],joint_fit,phiinner,joint_inner,joint=True)
            _,ridge_c,joint_w,rank_c,cond_c=best_c
            delta_c=joint_w[12:]
            pred_c=predict(phiq,delta_c.numpy());score_c,mse_c,_=skill(pred_c,true_query)
            zero_skill,zero_mse,_=skill(np.zeros_like(true_query),true_query)
            # D: explicitly privileged analytic constant-train-mean drag reference for U4.
            analytic_skill=analytic_mse=None
            if code==3:
              v=xquery[:,2:4]
              analytic=-DT*.5*train_b_mean*np.linalg.norm(v,axis=1,keepdims=True)*v
              analytic_skill,analytic_mse,_=skill(analytic,true_query)
            records.append({'regime':regime,'background_id':bg_id,'background_seed':bg_seed,
              'intervention':f'U{code+1}','fit_budget_per_law':budget,'query_probes':len(xquery),
              'A_separate_noisy_law_fit_skill':score_a,'A_mse':mse_a,'A_rank_before':rank_a,'A_rank_after':rank_a_after,
              'A_condition_before':cond_a,'A_condition_after':cond_a_after,
              'B_paired_projection_skill':score_b,'B_mse':mse_b,'B_selected_ridge':ridge_b,'B_rank':rank_b,'B_condition':cond_b,
              'C_joint_nuisance_cancel_skill':score_c,'C_mse':mse_c,'C_selected_ridge':ridge_c,'C_rank':rank_c,'C_condition':cond_c,
              'zero_effect_skill':zero_skill,'zero_effect_mse':zero_mse,
              'D_privileged_train_mean_analytic_skill':analytic_skill,'D_mse':analytic_mse,'true_effect_mean_square':signal})
            curves.append({'regime':regime,'background_id':bg_id,'intervention':f'U{code+1}','fit_budget_per_law':budget,
              'B_inner_ridge_scores':{str(r):float(m) for m,r,_,_,_ in candidates_b},
              'C_inner_ridge_scores':{str(r):float(m) for m,r,_,_,_ in candidates_c}})
    with (out/'per_background.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # Summaries and cluster bootstrap treat the diagnostic background as the unit.
    summary={}
    for regime in ('IDLIKE','BACKGROUND_OOD'):
      for code in ('U3','U4'):
       for budget in (128,512,1024):
        cell=[r for r in records if r['regime']==regime and r['intervention']==code and r['fit_budget_per_law']==budget]
        summary[f'{regime}|{code}|budget={budget}']={
          'n_backgrounds':len(cell),
          **{name:{'mean':float(np.mean([r[col] for r in cell])),
                   'median':float(np.median([r[col] for r in cell])),
                   'background_bootstrap_95pct_CI':bootstrap_ci([r[col] for r in cell],seed=args.seed+budget+code.__len__())}
             for name,col in [('A_separate_noisy_law_fit','A_separate_noisy_law_fit_skill'),
                              ('B_paired_projection','B_paired_projection_skill'),
                              ('C_joint_nuisance_cancel','C_joint_nuisance_cancel_skill'),
                              ('zero_effect','zero_effect_skill')]},
          'ratio_of_mean_error_to_mean_signal_skill':{
            name:float(1.-np.mean([r[mse_col] for r in cell])/np.mean([r['true_effect_mean_square'] for r in cell]))
            for name,mse_col in [('A_separate_noisy_law_fit','A_mse'),('B_paired_projection','B_mse'),
                                 ('C_joint_nuisance_cancel','C_mse'),('zero_effect','zero_effect_mse')]},
          'selected_ridge_B_by_background':[float(r['B_selected_ridge']) for r in cell],
          'selected_ridge_C_by_background':[float(r['C_selected_ridge']) for r in cell]}
    (out/'direct_effect_projection.json').write_text(json.dumps({'protocol':'privileged offline diagnostics only',
      'backgrounds_are_uncertainty_unit':True,'query_labels_used_for_fitting_or_ridge_selection':False,
      'metrics':summary,'ridge_grid':RIDGE_GRID,'per_fit_ridge_inner_scores':curves},indent=2),encoding='utf-8')
    train_sources={str((src/'basis_pretrained.pt').relative_to(ROOT)):sha(basis_ckpt),
      'scripts/q1_v2_direct_effect_projection.py':sha(Path(__file__))}
    (out/'provenance.json').write_text(json.dumps({'source_hashes':train_sources,'seed':args.seed,
      'test_split_used':False,'device':'cpu','rows':len(records),'runtime_seconds':None},indent=2),encoding='utf-8')
    print(f'Wrote paired projection diagnostic ({len(records)} rows) to {out}',flush=True)

if __name__=='__main__':main()
