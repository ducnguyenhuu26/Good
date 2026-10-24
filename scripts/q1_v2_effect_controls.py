"""Paired per-code query effects for support-conditioned, ablated, and zero effects."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from q1.datasets import stage_pair
from q1.simulator import PROGRAMS,DynamicsParams,apply_program,response
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import gaussian_posterior
from q1_v2.inference import composed
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import design
from scripts.q1_v2_pipeline import rows,tensor_input
from scripts.q1_v2_informative import scheduled,public

def skill(pred,true):
    d=float(np.mean(true**2));return 1-float(np.mean((pred-true)**2))/d if d>0 else None
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='results/q1/q1_v2_bayesian_operators_dev1');ap.add_argument('--informative',default='results/q1/q1_v2_informative_dev1');a=ap.parse_args();root=ROOT/a.run_dir;dest=ROOT/a.informative
    cfg=yaml.safe_load((root/'config.resolved.yaml').read_text(encoding='utf-8'));icfg=yaml.safe_load((ROOT/'configs/q1_v2_informative.yaml').read_text(encoding='utf-8'));manifest=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'));sched=json.loads((dest/'schedule_selection.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1);cp=torch.load(root/'basis_pretrained.pt',map_location='cpu',weights_only=False);basis=PartitionedBasis(hidden=cfg['hidden']);basis.load_state_dict(cp['basis']);basis.eval()
    for x in basis.parameters():x.requires_grad_(False)
    mean,cov,sigma2=cp['prior_mean'],cp['prior_covariance'],cp['sigma2'];cmap=json.loads((root/'operator_training.json').read_text(encoding='utf-8'))['opaque_code_map']
    aware={};ablated={}
    for seed in cfg['optimizer_seeds']:
      aware[seed]=AffineOperatorBank();aware[seed].load_state_dict(torch.load(root/f'operator_seed{seed}.pt',map_location='cpu',weights_only=False)['state_dict'])
      ablated[seed]=AffineOperatorBank();ablated[seed].load_state_dict(torch.load(root/f'context_ablated_operator_seed{seed}.pt',map_location='cpu',weights_only=False)['state_dict'])
    records=[]
    for regime in ('development_idlike','development_background_ood'):
      label='IDLIKE_DEVELOPMENT' if regime=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
      for bi,bg in enumerate(manifest[regime]['backgrounds']):
        params=DynamicsParams(np.asarray(bg['B']),bg['d'],bg['b'],bg['kappa']);seed=int(manifest[regime]['seeds'][bi])
        for code in range(4):
          program=(code,);after=apply_program(params,program)
          old,_,query=stage_pair(params,program,seed=seed+300000+PROGRAMS.index(program)*7919,n_support=cfg['old_support'],n_query=cfg['query_trajectory'],dt=cfg['dt'],sigma=cfg['sigma'])
          ox,oy=rows(old);qx,qy=rows(query);xq=tensor_input(qx);phi=design(basis,xq)
          oldpost=gaussian_posterior(design(basis,tensor_input(ox)),tensor_input(oy),mean,cov,sigma2)
          true=response(after,qx[:,:6],dt=cfg['dt'])-response(params,qx[:,:6],dt=cfg['dt'])
          for optseed in cfg['optimizer_seeds']:
            ac,bc=composed(aware[optseed],(cmap[str(code)],));apred=(phi@((ac@oldpost.mean+bc)-oldpost.mean)).reshape(-1,2).detach().numpy()
            fc,fd=composed(ablated[optseed],(cmap[str(code)],));fpred=(phi@((fc@mean+fd)-mean)).reshape(-1,2).detach().numpy()
            records.append({'protocol':'LOW_EXCITATION_PRIMARY','regime':label,'background_id':f'{label}-{bi:03d}','code':code,
              'optimizer_seed':optseed,'support_conditioned_skill':skill(apred,true),'context_ablated_skill':skill(fpred,true),'zero_effect_skill':0.0})
          # Same measured code under the frozen train-selected schedule.
          infseed=seed+icfg['seed'];oldi=scheduled(params,icfg['old_support'],infseed,sched['pattern'],icfg['hold_steps'],cfg['dt'],cfg['sigma'])
          qi=scheduled(after,icfg['query_trajectory'],infseed+500000+PROGRAMS.index(program)*7919,sched['pattern'],icfg['hold_steps'],cfg['dt'],cfg['sigma']);ix,iy=public(oldi);qx2,qy2=public(qi);phi2=design(basis,tensor_input(qx2));post2=gaussian_posterior(design(basis,tensor_input(ix)),tensor_input(iy),mean,cov,sigma2);truth2=response(after,qx2[:,:6],dt=cfg['dt'])-response(params,qx2[:,:6],dt=cfg['dt'])
          for optseed in cfg['optimizer_seeds']:
            ac,bc=composed(aware[optseed],(cmap[str(code)],));apred=(phi2@((ac@post2.mean+bc)-post2.mean)).reshape(-1,2).detach().numpy()
            fc,fd=composed(ablated[optseed],(cmap[str(code)],));fpred=(phi2@((fc@mean+fd)-mean)).reshape(-1,2).detach().numpy()
            records.append({'protocol':'INFORMATIVE_SECONDARY','regime':label,'background_id':f'{label}-{bi:03d}','code':code,
              'optimizer_seed':optseed,'support_conditioned_skill':skill(apred,truth2),'context_ablated_skill':skill(fpred,truth2),'zero_effect_skill':0.0})
    with (dest/'effect_controls.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary={}
    for r in records:summary.setdefault(f"{r['protocol']}|{r['regime']}|U{int(r['code'])+1}",[]).append(r)
    obj={k:{m:float(np.mean([x[m] for x in v])) for m in ('support_conditioned_skill','context_ablated_skill','zero_effect_skill')} for k,v in summary.items()}
    (dest/'effect_controls.json').write_text(json.dumps({'independent_query_skill':obj,'meaning':'effect skill is 1-MSE(predicted effect,true effect)/MSE(true effect); zero-effect baseline is exactly 0'},indent=2),encoding='utf-8')
    print(f'Wrote {len(records)} paired effect-control rows')
if __name__=='__main__':main()
