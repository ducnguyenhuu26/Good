"""Measure U3/U4 effect skill attainable in the frozen Q1 V2 basis."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch
from q1.datasets import rollout
from q1.simulator import DynamicsParams,apply_program,response
from q1_v2.basis import PartitionedBasis
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import design

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def transitions(traj):
    return np.c_[traj.states[:-1],traj.actions].astype(np.float32),traj.delta_v.astype(np.float32)

def fit_coefficients(basis,x,y):
    with torch.no_grad():
        phi=design(basis,torch.as_tensor(x,dtype=torch.float32))
        target=torch.as_tensor(y,dtype=torch.float32).reshape(-1)
        result=torch.linalg.lstsq(phi,target)
        w=result.solution
        singular=torch.linalg.svdvals(phi)
        rank=int(torch.linalg.matrix_rank(phi))
        cond=float(singular.max()/singular.min().clamp_min(1e-20))
    return w.detach(),rank,cond

def skill(pred,true):
    denom=float(np.mean(true**2))
    mse=float(np.mean((pred-true)**2))
    return (1.-mse/denom if denom>1e-20 else None),mse,denom

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-run',default='results/q1/q1_v2_operator_retrain_dev1')
    ap.add_argument('--output-dir',default='results/q1/q1_v2_frozen_basis_ceiling_dev1')
    ap.add_argument('--fit-transitions',type=int,default=768)
    ap.add_argument('--query-transitions-per-law',type=int,default=768)
    ap.add_argument('--transition-noise',type=float,default=0.0,
                    help='noise for both fit/query rollouts; zero isolates representational capacity from label noise')
    args=ap.parse_args()
    if not 512<=args.fit_transitions<=1024: raise ValueError('fit transitions must be in [512,1024]')
    if args.query_transitions_per_law<512: raise ValueError('query trajectories must be independent and at least 512 transitions')
    src=ROOT/args.source_run;out=ROOT/args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'refusing to overwrite existing outputs: {out}')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    cfg=json.loads((src/'provenance.json').read_text(encoding='utf-8'))
    resolved=__import__('yaml').safe_load((src/'config.resolved.yaml').read_text(encoding='utf-8'))
    manifest=json.loads((src/'split_manifest.json').read_text(encoding='utf-8'))
    basis_data=torch.load(src/'basis_pretrained.pt',map_location='cpu',weights_only=False)
    basis=PartitionedBasis(hidden=int(resolved['hidden'])).cpu()
    basis.load_state_dict(basis_data['basis']);basis.eval()
    for p in basis.parameters():p.requires_grad_(False)
    bank_info={}
    for opt_seed in resolved['optimizer_seeds']:
        saved=torch.load(src/f'operator_seed{opt_seed}.pt',map_location='cpu',weights_only=False)
        bank=AffineOperatorBank().cpu();bank.load_state_dict(saved['state_dict']);bank.eval()
        for p in bank.parameters():p.requires_grad_(False)
        bank_info[int(opt_seed)]=(bank,{int(k):int(v) for k,v in saved['opaque_code_map'].items()})
    rows=[];coefficient_records={}
    regimes=[('IDLIKE_DEVELOPMENT','development_idlike'),('BACKGROUND_OOD_DEVELOPMENT','development_background_ood')]
    fit_seed_records={};query_seed_records={}
    for regime,manifest_key in regimes:
        for bi,bg in enumerate(manifest[manifest_key]['backgrounds']):
            params=DynamicsParams(np.asarray(bg['B']),float(bg['d']),float(bg['b']),float(bg['kappa']))
            bg_seed=int(manifest[manifest_key]['seeds'][bi]);bg_id=f'{regime}-{bi:03d}'
            before_fit_seed=bg_seed+1100000
            before_traj=rollout(params,n=args.fit_transitions,seed=before_fit_seed,dt=resolved['dt'],sigma=args.transition_noise)
            before_x,before_y=transitions(before_traj)
            w_before,rank_before,cond_before=fit_coefficients(basis,before_x,before_y)
            coefficient_records[bg_id]={'before':w_before.tolist(),'rank_before':rank_before,'condition_before':cond_before}
            for code in (2,3):
                program=(code,);after=apply_program(params,program)
                fit_seed=bg_seed+1200000+code*10007
                after_traj=rollout(after,n=args.fit_transitions,seed=fit_seed,dt=resolved['dt'],sigma=args.transition_noise)
                after_x,after_y=transitions(after_traj)
                w_after,rank_after,cond_after=fit_coefficients(basis,after_x,after_y)
                query_before_seed=bg_seed+1300000+code*10007
                query_after_seed=bg_seed+1400000+code*10007
                query_before=rollout(params,n=args.query_transitions_per_law,seed=query_before_seed,dt=resolved['dt'],sigma=args.transition_noise)
                query_after=rollout(after,n=args.query_transitions_per_law,seed=query_after_seed,dt=resolved['dt'],sigma=args.transition_noise)
                qx=np.concatenate((transitions(query_before)[0],transitions(query_after)[0]),axis=0)
                true_effect=response(after,qx,dt=resolved['dt'])-response(params,qx,dt=resolved['dt'])
                with torch.no_grad():
                    qphi=design(basis,torch.as_tensor(qx,dtype=torch.float32))
                    basis_effect=(qphi@(w_after-w_before)).reshape(-1,2).cpu().numpy()
                basis_skill,basis_mse,signal=skill(basis_effect,true_effect)
                row={'regime':regime,'background_id':bg_id,'background_seed':bg_seed,
                    'intervention':f'U{code+1}','fit_transitions_each_law':args.fit_transitions,
                    'query_transitions':len(qx),'query_before_seed':query_before_seed,'query_after_seed':query_after_seed,
                    'design_rank_before':rank_before,'design_rank_after':rank_after,
                    'design_condition_before':cond_before,'design_condition_after':cond_after,
                    'true_effect_mean_square':signal,'frozen_basis_ceiling_skill':basis_skill,
                    'frozen_basis_ceiling_mse':basis_mse,'zero_effect_skill':0.0,
                    'zero_effect_mse':signal}
                for opt_seed,(bank,opaque_map) in bank_info.items():
                    opaque_code=opaque_map[code]
                    am=bank.matrix(opaque_code);bb=bank.bias[opaque_code]
                    operator_effect=(qphi@((am@w_before+bb)-w_before)).reshape(-1,2).cpu().numpy()
                    op_skill,op_mse,_=skill(operator_effect,true_effect)
                    row[f'operator_seed{opt_seed}_skill']=op_skill
                    row[f'operator_seed{opt_seed}_mse']=op_mse
                rows.append(row)
                coefficient_records[bg_id][f'U{code+1}_after']=w_after.tolist()
                coefficient_records[bg_id][f'U{code+1}_after_rank']=rank_after
                coefficient_records[bg_id][f'U{code+1}_after_condition']=cond_after
                fit_seed_records[f'{bg_id}|U{code+1}']={'before':before_fit_seed,'after':fit_seed}
                query_seed_records[f'{bg_id}|U{code+1}']={'before':query_before_seed,'after':query_after_seed}
    with (out/'per_background.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'coefficients.json').write_text(json.dumps(coefficient_records,indent=2),encoding='utf-8')
    summary={}
    for regime,_ in regimes:
        for intervention in ('U3','U4'):
            cell=[r for r in rows if r['regime']==regime and r['intervention']==intervention]
            vals={'n_backgrounds':len(cell),'frozen_basis_ceiling_skill_mean':float(np.mean([r['frozen_basis_ceiling_skill'] for r in cell])),
                'frozen_basis_ceiling_skill_median':float(np.median([r['frozen_basis_ceiling_skill'] for r in cell])),
                'zero_effect_skill':0.0,'frozen_basis_ceiling_mse_mean':float(np.mean([r['frozen_basis_ceiling_mse'] for r in cell]))}
            for opt_seed in bank_info:
                skills=[r[f'operator_seed{opt_seed}_skill'] for r in cell]
                vals[f'operator_seed{opt_seed}_skill_mean']=float(np.mean(skills))
                vals[f'operator_seed{opt_seed}_skill_median']=float(np.median(skills))
            vals['operator_skill_mean_over_seeds']=float(np.mean([r[f'operator_seed{s}_skill'] for r in cell for s in bank_info]))
            summary[f'{regime}|{intervention}']=vals
    source_checkpoint=src/'basis_pretrained.pt'
    result={'status':'FRESH_DEVELOPMENT_DIAGNOSTIC','basis_frozen':True,
        'basis_checkpoint_sha256':sha(source_checkpoint),'fit_transitions_per_law':args.fit_transitions,
        'transition_noise':args.transition_noise,
        'independent_query_transitions':args.query_transitions_per_law*2,
        'query_protocol':'independent trajectories under before and after laws; simulator response difference scored on the pooled x inputs',
        'coefficient_fit':'unregularized least squares using frozen Phi and observed transitions',
        'operator_prediction':'saved operator matrix and bias applied to the 768-transition before-law coefficient estimate',
        'development_seed_sets_reused':True,'test_split_used':False,'metrics':summary,
        'seed_provenance':{'fit':fit_seed_records,'query':query_seed_records}}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Frozen V2 basis representational ceiling','',
      f"Frozen basis checkpoint SHA-256: `{result['basis_checkpoint_sha256']}`. Coefficients for each law were fit with {args.fit_transitions} transitions; effect skill was scored on {args.query_transitions_per_law*2} independent transitions per background/intervention with transition noise {args.transition_noise}. No test split was used.",'',
      '| Regime | Effect | Frozen-basis ceiling | Learned operators | Zero-effect | Backgrounds |','|---|---|---:|---:|---:|---:|']
    for k,v in summary.items():
        regime,intervention=k.split('|');lines.append(f"| {regime} | {intervention} | {v['frozen_basis_ceiling_skill_mean']:.3f} | {v['operator_skill_mean_over_seeds']:.3f} | 0.000 | {v['n_backgrounds']} |")
    lines+=['','Skill is `1 - query MSE / mean(true effect squared)`; zero means no gain over predicting no effect, and negative values are worse. The basis estimate uses separately fit before/after coefficients and is an empirical frozen-representation ceiling under this excitation and sample budget. The operator prediction uses each saved V2 bank with the same frozen basis and the before-law coefficient fit. Noise-free fitting isolates representation and operator error from transition measurement noise.','']
    (out/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps({'source_run':args.source_run,'source_checkpoint_sha256':sha(source_checkpoint),
        'operator_checkpoint_sha256':{str(s):sha(src/f'operator_seed{s}.pt') for s in bank_info},
        'script_sha256':sha(Path(__file__)),'fit_transitions':args.fit_transitions,'query_transitions_per_law':args.query_transitions_per_law,'transition_noise':args.transition_noise,
        'rows':len(rows),'torch':torch.__version__,'device':'cpu','test_split_used':False},indent=2),encoding='utf-8')
    print(f"Wrote {len(rows)} background/intervention rows to {out}")

if __name__=='__main__':main()
