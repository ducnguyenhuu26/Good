"""Paired legacy M3/M4 evaluation under the train-selected informative schedule."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS,DynamicsParams
from q1.evaluate import _public_transitions,_support_tensor
from q1.inference import rank_programs
from q1.primitives import PrimitiveBank
from q1.train_primitives import _get_m2
from scripts.q1_v2_informative import scheduled,public

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='results/q1/q1_v2_bayesian_operators_dev1');p.add_argument('--output',default='results/q1/q1_v2_informative_dev1');a=p.parse_args();root=ROOT/a.run_dir;out=ROOT/a.output
    cfg=yaml.safe_load((root/'config.resolved.yaml').read_text(encoding='utf-8'));icfg=yaml.safe_load((ROOT/'configs/q1_v2_informative.yaml').read_text(encoding='utf-8'));man=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'));selected=json.loads((out/'schedule_selection.json').read_text(encoding='utf-8'));pattern=selected['pattern']
    device=torch.device('cpu');torch.set_num_threads(1);m2path=ROOT/'results/q1/cpu_confirmatory_impl/m2/context_model.pt';m3path=ROOT/'results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt';m4path=ROOT/'results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt'
    m2,norm,m2data=_get_m2(m2path,device);d3=torch.load(m3path,map_location=device,weights_only=False);d4=torch.load(m4path,map_location=device,weights_only=False);cmap=d4['group_to_code'];inv={int(v):int(k) for k,v in cmap.items()};dim=int(m2data['config']['model']['context_dim'])
    banks=[]
    for data in (d3,d4):
        b=PrimitiveBank(dim,8,64);b.load_state_dict(data['primitives']);b.eval();banks.append((b,float(d4['length_penalty'])))
    rows_out=[]
    for reg in ('development_idlike','development_background_ood'):
      label='IDLIKE_DEVELOPMENT' if reg=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
      for bi,bg in enumerate(man[reg]['backgrounds']):
        before=DynamicsParams(np.asarray(bg['B']),bg['d'],bg['b'],bg['kappa']);cl,_=functional_classes(before,dt=cfg['dt']);seed=int(man[reg]['seeds'][bi])+icfg['seed']
        for pi,tp in enumerate(PROGRAMS):
          old=scheduled(before,icfg['old_support'],seed,pattern,icfg['hold_steps'],cfg['dt'],cfg['sigma']);after=__import__('q1.simulator',fromlist=['apply_program']).apply_program(before,tp)
          adapt=scheduled(after,icfg['adaptation_trajectory'],seed+pi*7919,pattern,icfg['hold_steps'],cfg['dt'],cfg['sigma'],initial_state=old.states[-1],start_time=icfg['old_support'])
          query=scheduled(after,icfg['query_trajectory'],seed+500000+pi*7919,pattern,icfg['hold_steps'],cfg['dt'],cfg['sigma']);ctx=m2.encode(_support_tensor(old.states,old.actions,norm,device));ax,ay=public(adapt);ar=np.c_[ax,ay];qr=public(query);x=torch.as_tensor(qr[0],dtype=torch.float32);y=torch.as_tensor(qr[1],dtype=torch.float32)
          for n in cfg['budgets']:
            for model,(bank,penalty) in zip(('M3','M4'),banks):
              ordered,_=rank_programs(ctx,torch.as_tensor(ar,dtype=torch.float32)[:n],encoder=m2.encoder,executor=m2.executor,primitives=bank,normalizer=norm,length_penalty=penalty,device=device);codep=ordered[0];predp=tuple(inv[c] for c in codep)
              with torch.no_grad():qpred=norm.denormalize_dv(m2.executor(norm.normalize_x(x),bank.apply_program(ctx,codep)));mse=float(torch.mean((qpred-y)**2))
              rows_out.append({'regime':label,'background_id':f'{label}-{bi:03d}','true_program':json.dumps(tp),'N':n,'model':model,'predicted_program':json.dumps(predp),'functional_class_correct':int(cl[PROGRAMS.index(predp)]==cl[pi]),'query_delta_v_mse':mse})
    with (out/'legacy_per_background.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows_out[0]));w.writeheader();w.writerows(rows_out)
    summary={}
    for r in rows_out:summary.setdefault(f"{r['regime']}|{r['model']}|N={r['N']}",[]).append(r)
    metrics={k:{'functional_class_accuracy':float(np.mean([x['functional_class_correct'] for x in v])),'query_delta_v_mse':float(np.mean([x['query_delta_v_mse'] for x in v])),'case_count':len(v)} for k,v in summary.items()}
    (out/'legacy_metrics.json').write_text(json.dumps({'same_informative_trajectories_as_v2':True,'metrics':metrics,'legacy_checkpoints':{'m2':str(m2path.relative_to(ROOT)),'m3':str(m3path.relative_to(ROOT)),'m4':str(m4path.relative_to(ROOT))}},indent=2),encoding='utf-8')
    print(f'Wrote {len(rows_out)} paired informative legacy rows')
if __name__=='__main__':main()
