"""Train/inner-only coefficient mismatch and shrinkage audit for root repair."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from q1.datasets import rollout
from q1.simulator import apply_program,sample_params
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import gaussian_posterior
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import design
from scripts.q1_v2_pipeline import rows
from scripts.q1_v2_root_repair import TRAIN_COUNTS,crossfit_effect_tasks

def main():
    out=ROOT/'results/q1/q1_v2_root_repair_dev3';torch.set_num_threads(1)
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    tdata=json.loads((out/'operator_training.json').read_text(encoding='utf-8'))
    cp=torch.load(out/'basis_pretrained.pt',map_location='cpu',weights_only=False)
    basis=PartitionedBasis(widths=tuple(cp['widths']),hidden=32);basis.load_state_dict(cp['basis']);basis.eval()
    bankdata=torch.load(out/'operator_seed4201_effect_aware.pt',map_location='cpu',weights_only=False)
    bank=AffineOperatorBank(widths=tuple(cp['widths']));bank.load_state_dict(bankdata['state_dict']);bank.eval()
    code_map={int(k):int(v) for k,v in tdata['opaque_code_map'].items()}
    prior_mean=cp['prior_mean'];prior_cov=cp['prior_covariance'];sigma2=float(cp['sigma2'])
    all_rows=[]
    for split,key in [('train','new_train'),('inner_train','new_inner_train')]:
        seeds=[int(s) for s in protocol[key]['seeds']]
        params=[sample_params(np.random.default_rng(s),'train',TRAIN_COUNTS) for s in seeds]
        tasks,_,_=crossfit_effect_tasks(params,seeds,split,basis,code_map=code_map)
        for task in tasks:
            physical=next(c for c,v in code_map.items() if v==int(task.opaque_code))
            bi=int(task.background_id.rsplit('-',1)[1]);bg_seed=seeds[bi]
            before=sample_params(np.random.default_rng(bg_seed),'train',TRAIN_COUNTS)
            after=apply_program(before,(physical,))
            oldtr=rollout(before,n=96,seed=bg_seed+5100000+physical*1009,dt=.05,sigma=.0005)
            newtr=rollout(after,n=96,seed=bg_seed+5200000+physical*1009,dt=.05,sigma=.0005)
            old_x,old_y=rows(oldtr);new_x,new_y=rows(newtr)
            with torch.no_grad():
                oldpost=gaussian_posterior(design(basis,torch.as_tensor(old_x)),torch.as_tensor(old_y),prior_mean,prior_cov,sigma2)
                postpost=gaussian_posterior(design(basis,torch.as_tensor(new_x)),torch.as_tensor(new_y),prior_mean,prior_cov,sigma2)
                a=bank.matrix(int(task.opaque_code));b=bank.bias[int(task.opaque_code)]
                identity=torch.eye(len(oldpost.mean))
                observed_delta=task.delta_coeff
                posterior_delta=(a-identity)@oldpost.mean+b
                learned_fit_delta=(a-identity)@task.before_coeff+b
                probe=design(basis,task.probe_x)
                true_effect=probe@observed_delta
                pred_effect=probe@posterior_delta
                fit_effect=probe@learned_fit_delta
                effect_energy=float(true_effect.square().mean())
                posterior_skill=1-float((pred_effect-true_effect).square().mean())/max(effect_energy,1e-30)
                fit_skill=1-float((fit_effect-true_effect).square().mean())/max(effect_energy,1e-30)
                coeff_energy=float(observed_delta.square().mean())
                posterior_coeff_skill=1-float((posterior_delta-observed_delta).square().mean())/max(coeff_energy,1e-30)
                fit_coeff_skill=1-float((learned_fit_delta-observed_delta).square().mean())/max(coeff_energy,1e-30)
            before_fit=task.before_coeff;after_fit=before_fit+task.delta_coeff
            all_rows.append({'split':split,'background_id':task.background_id,'background_seed':bg_seed,
              'physical_code':physical,'opaque_code':int(task.opaque_code),'fold':task.fold,
              'old_posterior_coeff_norm':float(oldpost.mean.norm()),
              'old_crossfit_fit_coeff_norm':float(before_fit.norm()),
              'old_posterior_vs_crossfit_relative_error':float((oldpost.mean-before_fit).norm()/before_fit.norm().clamp_min(1e-12)),
              'independent_post_fit_coeff_norm':float(after_fit.norm()),
              'separate_fit_delta_coeff_norm':float(observed_delta.norm()),
              'separate_fit_post_posterior_coeff_norm':float(postpost.mean.norm()),
              'posterior_operator_delta_coeff_norm':float(posterior_delta.norm()),
              'posterior_operator_to_separate_delta_norm_ratio':float(posterior_delta.norm()/observed_delta.norm().clamp_min(1e-12)),
              'posterior_function_effect_skill_on_crossfit_probe':posterior_skill,
              'posterior_coefficient_delta_skill':posterior_coeff_skill,
              'crossfit_before_operator_delta_coeff_norm':float(learned_fit_delta.norm()),
              'crossfit_operator_to_separate_delta_norm_ratio':float(learned_fit_delta.norm()/observed_delta.norm().clamp_min(1e-12)),
              'crossfit_function_effect_skill_on_crossfit_probe':fit_skill,
              'crossfit_coefficient_delta_skill':fit_coeff_skill,
              'posterior_learned_effect_energy':float(pred_effect.square().mean()),
              'crossfit_learned_effect_energy':float(fit_effect.square().mean()),
              'observed_crossfit_effect_energy':effect_energy})
    import csv
    with (out/'coefficient_diagnostics_per_background.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    summary={}
    for split in ('train','inner_train'):
      for physical in range(4):
        cell=[r for r in all_rows if r['split']==split and r['physical_code']==physical]
        summary[f'{split}|U{physical+1}']={'n_fold_rows':len(cell),'n_backgrounds':len({r['background_id'] for r in cell}),
          **{k:float(np.mean([r[k] for r in cell])) for k in ('old_posterior_vs_crossfit_relative_error',
            'posterior_operator_to_separate_delta_norm_ratio','posterior_function_effect_skill_on_crossfit_probe',
            'posterior_coefficient_delta_skill','crossfit_operator_to_separate_delta_norm_ratio',
            'crossfit_function_effect_skill_on_crossfit_probe','crossfit_coefficient_delta_skill',
            'posterior_learned_effect_energy','crossfit_learned_effect_energy','observed_crossfit_effect_energy')}}
    (out/'coefficient_diagnostics.json').write_text(json.dumps({'information':'noisy observed transitions only; train and inner-train backgrounds; no conditional-mean response labels','optimizer_seed':4201,'summary':summary},indent=2),encoding='utf-8')
    print(f'Wrote train/inner coefficient audit ({len(all_rows)} fold rows)')
if __name__=='__main__':main()
