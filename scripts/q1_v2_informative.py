"""Secondary informative-exploration evaluation; schedule selected on train only."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from q1.simulator import PROGRAMS,DynamicsParams,apply_program,sample_params,transition,response
from q1.oracle import functional_classes
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import gaussian_posterior,GaussianPosterior
from q1_v2.inference import rank_programs
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import design
from scripts.q1_v2_pipeline import rows,tensor_input

class Traj:
    def __init__(self,s,a):self.states=s;self.actions=a
    @property
    def delta_v(self):return self.states[1:,2:]-self.states[:-1,2:]

def scheduled(params,n,seed,pattern,hold,dt,sigma,initial_state=None,start_time=0):
    ir=np.random.default_rng(seed);nr=np.random.default_rng(seed+209759)
    if initial_state is None:state=ir.uniform(-.25,.25,4);state[2:]=ir.uniform(-.65,.65,2)
    else:state=np.asarray(initial_state,float).copy()
    states=np.empty((n+1,4));actions=np.empty((n,2));states[0]=state
    for t in range(n):
        idx=((start_time+t)//hold)%len(pattern);actions[t]=np.asarray(pattern[idx],float)
        states[t+1]=transition(states[t],actions[t],params,dt=dt,sigma=sigma,rng=nr)
        if np.linalg.norm(states[t+1])>100:raise FloatingPointError('unstable informative trajectory')
    return Traj(states,actions)

def public(t):return np.c_[t.states[:-1],t.actions].astype(np.float32),t.delta_v.astype(np.float32)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='results/q1/q1_v2_bayesian_operators_dev1');ap.add_argument('--output',default='results/q1/q1_v2_informative_dev1');a=ap.parse_args();out=ROOT/a.run_dir;dest=ROOT/a.output
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
    dest.mkdir(parents=True,exist_ok=True);cfg=yaml.safe_load((out/'config.resolved.yaml').read_text(encoding='utf-8'));icfg=yaml.safe_load((ROOT/'configs/q1_v2_informative.yaml').read_text(encoding='utf-8'));manifest=json.loads((out/'split_manifest.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1);basis_data=torch.load(out/'basis_pretrained.pt',map_location='cpu',weights_only=False);basis=PartitionedBasis(hidden=cfg['hidden']);basis.load_state_dict(basis_data['basis']);basis.eval()
    for p in basis.parameters():p.requires_grad_(False)
    mean,cov,sigma2=basis_data['prior_mean'],basis_data['prior_covariance'],basis_data['sigma2'];cmap=json.loads((out/'operator_training.json').read_text(encoding='utf-8'))['opaque_code_map']
    seeds=cfg['optimizer_seeds'];banks={};free={}
    for s in seeds:
        banks[s]=AffineOperatorBank();banks[s].load_state_dict(torch.load(out/f'operator_seed{s}.pt',map_location='cpu',weights_only=False)['state_dict']);banks[s].eval()
        free[s]=AffineOperatorBank();free[s].load_state_dict(torch.load(out/f'context_ablated_operator_seed{s}.pt',map_location='cpu',weights_only=False)['state_dict']);free[s].eval()
        for b in (banks[s],free[s]):
            for p in b.parameters():p.requires_grad_(False)
    # Choose among the fixed schedules using only eight train-range backgrounds and old data.
    train_bgs=manifest['train']['backgrounds'][:int(icfg['training_backgrounds_for_schedule_selection'])]
    scores={name:[] for name in icfg['schedule_candidates']}
    for name,pattern in icfg['schedule_candidates'].items():
      for i,bg in enumerate(train_bgs):
        params=DynamicsParams(np.asarray(bg['B']),bg['d'],bg['b'],bg['kappa']);tr=scheduled(params,96,icfg['seed']+i*1009,pattern,icfg['hold_steps'],cfg['dt'],cfg['sigma'])
        x=np.c_[tr.states[:-1],tr.actions];covx=np.cov(x,rowvar=False)+1e-6*np.eye(6);covv=np.cov(tr.states[:-1,2:].T)+1e-6*np.eye(2)
        score=np.linalg.slogdet(covx)[1]+0.25*np.linalg.slogdet(covv)[1]
        scores[name].append({'coverage_score':float(score),'max_state_norm':float(np.linalg.norm(tr.states,axis=1).max()),'velocity_spread':float(np.trace(covv))})
    valid=[n for n in scores if max(r['max_state_norm'] for r in scores[n])<=icfg['state_norm_bound']]
    if not valid:raise RuntimeError('no predeclared schedule passes training-only bounded-state check')
    selected=max(valid,key=lambda n:np.mean([r['coverage_score'] for r in scores[n]]));pattern=icfg['schedule_candidates'][selected]
    schedule_info={'selected':selected,'candidates':scores,'selection':'training-only state-action logdet plus velocity spread; state bound checked on training backgrounds','pattern':pattern,'hold_steps':icfg['hold_steps']}
    (dest/'schedule_selection.json').write_text(json.dumps(schedule_info,indent=2),encoding='utf-8')
    rows_out=[];effect_rows=[];started=time.time()
    for regime in ('development_idlike','development_background_ood'):
      label='IDLIKE_DEVELOPMENT' if regime=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
      bgs=manifest[regime]['backgrounds'];bseed=manifest[regime]['seeds']
      for bi,bg in enumerate(bgs):
        print(f'Informative {label} background {bi+1}/{len(bgs)}',flush=True)
        params=DynamicsParams(np.asarray(bg['B']),bg['d'],bg['b'],bg['kappa']);classes,_=functional_classes(params,dt=cfg['dt'])
        sid=int(bseed[bi])+icfg['seed'];old=scheduled(params,icfg['old_support'],sid,pattern,icfg['hold_steps'],cfg['dt'],cfg['sigma'])
        ox,oy=public(old);oldpost_by_seed={s:gaussian_posterior(design(basis,tensor_input(ox)),tensor_input(oy),mean,cov,sigma2) for s in seeds}
        for pi,tp in enumerate(PROGRAMS):
          after=apply_program(params,tp);adapt=scheduled(after,icfg['adaptation_trajectory'],sid+pi*7919,pattern,icfg['hold_steps'],cfg['dt'],cfg['sigma'],initial_state=old.states[-1],start_time=icfg['old_support'])
          q=scheduled(after,icfg['query_trajectory'],sid+500000+pi*7919,pattern,icfg['hold_steps'],cfg['dt'],cfg['sigma']);ax,ay=public(adapt);qx,qy=public(q);qphi=design(basis,tensor_input(qx));qtarget=tensor_input(qy).reshape(-1)
          for n in icfg['budgets']:
            code_candidates=[tuple(cmap[str(c)] for c in p) for p in PROGRAMS]
            for optseed in seeds:
              ranked=rank_programs(banks[optseed],oldpost_by_seed[optseed],code_candidates,design(basis,tensor_input(ax[:n])),tensor_input(ay[:n]),sigma2,0.0)
              inv={int(v):int(k) for k,v in cmap.items()};predp=tuple(inv[c] for c in ranked[0][0]);predmean=ranked[0][2]
              from q1_v2.inference import composed
              aa,bb=composed(banks[optseed],ranked[0][0]);pm=aa@oldpost_by_seed[optseed].mean+bb
              qmse=float(torch.mean(((qphi@pm).reshape(-1)-qtarget)**2).detach())
              rows_out.append({'regime':label,'background_id':f'{label}-{bi:03d}','true_program':json.dumps(tp),'N':n,'optimizer_seed':optseed,'predicted_program':json.dumps(predp),'functional_class_correct':int(classes[PROGRAMS.index(predp)]==classes[pi]),'query_delta_v_mse':qmse,'old_posterior_trace':float(torch.trace(oldpost_by_seed[optseed].covariance))})
          if len(tp)==1:
            for optseed in seeds:
              aa,bb=composed(banks[optseed],(cmap[str(tp[0])],));pred=(qphi@((aa@oldpost_by_seed[optseed].mean+bb)-oldpost_by_seed[optseed].mean)).reshape(-1,2).detach().numpy()
              true=response(after,qx[:,:6],dt=cfg['dt'])-response(params,qx[:,:6],dt=cfg['dt']);den=float(np.mean(true**2))
              effect_rows.append({'regime':label,'background_id':f'{label}-{bi:03d}','code':tp[0],'optimizer_seed':optseed,'effect_skill':1-float(np.mean((pred-true)**2))/den if den>0 else None})
    for fn,data in [('per_background.csv',rows_out),('per_code_effect.csv',effect_rows)]:
      with (dest/fn).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary={}
    for r in rows_out:summary.setdefault(f"{r['regime']}|N={r['N']}",[]).append(r)
    metrics={k:{'functional_class_accuracy':float(np.mean([x['functional_class_correct'] for x in v])),'query_delta_v_mse':float(np.mean([x['query_delta_v_mse'] for x in v])),'mean_old_posterior_trace':float(np.mean([x['old_posterior_trace'] for x in v])),'case_count':len(v)} for k,v in summary.items()}
    (dest/'metrics.json').write_text(json.dumps({'status':'SECONDARY_DEVELOPMENT_DIAGNOSTIC','selected_schedule':selected,'metrics':metrics,'runtime_seconds':time.time()-started,'test_split_used':False,'limitations':['Legacy M4 has not yet been evaluated under this informative schedule.','No confirmatory/test split was used.']},indent=2),encoding='utf-8')
    print(f'Wrote informative secondary development metrics: {dest}',flush=True)
if __name__=='__main__':main()
