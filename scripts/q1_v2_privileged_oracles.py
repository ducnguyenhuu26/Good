"""Offline known-old and old-support system-ID references; never learner inputs."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from q1.datasets import rollout,stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS,DynamicsParams
from q1.diagnostics.feasibility_abc import estimate_old_params,score_candidates,parameter_vector

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='results/q1/q1_v2_bayesian_operators_dev1');a=p.parse_args();out=ROOT/a.run_dir
    cfg=__import__('yaml').safe_load((out/'config.resolved.yaml').read_text(encoding='utf-8'));m=json.loads((out/'split_manifest.json').read_text(encoding='utf-8'))
    records=[];fitrows=[];start=time.time()
    for reg in ('development_idlike','development_background_ood'):
        label='IDLIKE_DEVELOPMENT' if reg=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
        for bi,bg in enumerate(m[reg]['backgrounds']):
            before=DynamicsParams(np.asarray(bg['B']),float(bg['d']),float(bg['b']),float(bg['kappa']))
            old=rollout(before,n=cfg['old_support'],seed=int(m[reg]['seeds'][bi])+300000,dt=cfg['dt'],sigma=cfg['sigma'])
            estimated,meta=estimate_old_params(old.states[:-1],old.actions,old.delta_v,dt=cfg['dt'],
                seed=int(m[reg]['seeds'][bi])+4700,max_nfev=400,initializations=8)
            truth=parameter_vector(before);hat=parameter_vector(estimated)
            fitrows.append({'regime':label,'background_id':f'{label}-{bi:03d}',
                'old_support_rmse':meta['old_support_rmse'],'parameter_rmse':float(np.sqrt(np.mean((truth-hat)**2))),
                'b_error':float(hat[5]-truth[5]),'condition_number':meta['jacobian_condition_number']})
            cls,_=functional_classes(before,dt=cfg['dt']);base_seed=int(m[reg]['seeds'][bi])
            for pi,program in enumerate(PROGRAMS):
                _,adapt,_=stage_pair(before,program,seed=base_seed+300000+pi*7919,
                    n_support=cfg['old_support'],n_query=cfg['adaptation_trajectory'],dt=cfg['dt'],sigma=cfg['sigma'])
                probes=np.c_[adapt.states[:-1],adapt.actions];observed=adapt.delta_v
                for n in cfg['budgets']:
                    truth_scores=score_candidates(before,probes[:n],observed[:n],dt=cfg['dt'])
                    est_scores=score_candidates(estimated,probes[:n],observed[:n],dt=cfg['dt'])
                    for name,scores in [('known_old_privileged',truth_scores),('estimated_old_privileged',est_scores)]:
                        pred=PROGRAMS[int(np.argmin(scores))]
                        records.append({'regime':label,'background_id':f'{label}-{bi:03d}',
                          'true_program':json.dumps(program),'N':n,'oracle':name,
                          'functional_class_correct':int(cls[PROGRAMS.index(pred)]==cls[pi]),
                          'predicted_program':json.dumps(pred),
                          'true_program_mse':float(scores[pi]),'best_score':float(np.min(scores))})
    for filename,rows in [('privileged_oracle_per_background.csv',records),('privileged_old_fit.csv',fitrows)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'privileged_oracle_provenance.json').write_text(json.dumps({'label':'PRIVILEGED_DIAGNOSTIC_ONLY',
        'known_old_uses_true_background_params_for_reference_scoring':True,
        'estimated_old_fit_scope':'only the 96 observed old-stage transitions; generic existing bounded system-ID diagnostic',
        'neither_oracle_is_a_v2_model_input':True,'max_nfev':400,'multistarts':8,
        'runtime_seconds':time.time()-start},indent=2),encoding='utf-8')
    print(f'Wrote {len(records)} privileged oracle rows')
if __name__=='__main__':main()
