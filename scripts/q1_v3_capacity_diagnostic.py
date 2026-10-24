"""Offline matched-input representability check; privileged targets only here."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from scipy.optimize import least_squares
from q1.simulator import DynamicsParams,apply_program,response
from q1_v3.mechanisms import FixedTanhAction,OddActionResponse,mean_delta_v

def probes(seed,n):
    r=np.random.default_rng(seed);x=np.c_[r.uniform(-.7,.7,(n,2)),r.uniform(-.5,.5,(n,2)),r.uniform(-1,1,(n,2))]
    return torch.as_tensor(x,dtype=torch.float64),x
def fit_delta(theta,x,y,action,dt):
    base=theta.detach().numpy()
    with torch.no_grad():base_mean=mean_delta_v(x,theta,action,dt).reshape(-1)
    def residual(z):
        with torch.no_grad():pred=mean_delta_v(x,torch.as_tensor(z,dtype=torch.float64),action,dt).reshape(-1)
        return pred.numpy()-base_mean.numpy()-y.detach().numpy().reshape(-1)
    lo=np.array([-2.5]*4+[-6.]*3);hi=np.array([2.5]*4+[1.]*3)
    return least_squares(residual,np.clip(base,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10)
def main():
    run=ROOT/'results/q1/q1_v3_mechanism_dev1';cfg=yaml.safe_load((run/'config.resolved.yaml').read_text(encoding='utf-8'))
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'));out=[]
    ck=torch.load(run/'action_learned_odd.pt',map_location='cpu',weights_only=False);learned=OddActionResponse().double();learned.load_state_dict(ck['state_dict']);learned.eval()
    tanh=FixedTanhAction();split=protocol['split_manifest']
    for reg,key in [('IDLIKE_DEVELOPMENT','development_idlike'),('BACKGROUND_OOD_DEVELOPMENT','development_background_ood')]:
      for bi,bg in enumerate(split[key]['backgrounds']):
        p=DynamicsParams(np.asarray(bg['B']),float(bg['d']),float(bg['b']),float(bg['kappa']))
        theta=torch.tensor([*p.B.reshape(-1),np.log(p.d),np.log(p.b),np.log(p.kappa)],dtype=torch.float64)
        xf_npfit=probes(int(split[key]['seeds'][bi])+7001,1024);xq_t,xq_np=probes(int(split[key]['seeds'][bi])+7002,1024);xf=xf_npfit[0]
        for variant,action in [('learned_odd',learned),('fixed_tanh_diagnostic',tanh)]:
          for code in range(4):
            after=apply_program(p,(code,));yf_np=response(after,xf_npfit[1],dt=cfg['dt'])-response(p,xf_npfit[1],dt=cfg['dt']);yf=torch.as_tensor(yf_np,dtype=torch.float64)
            optimized=fit_delta(theta,xf,yf,action,cfg['dt']);after_theta=torch.as_tensor(optimized.x,dtype=torch.float64)
            with torch.no_grad():pred=mean_delta_v(xq_t,after_theta,action,cfg['dt'])-mean_delta_v(xq_t,theta,action,cfg['dt'])
            truth=response(after,xq_np,dt=cfg['dt'])-response(p,xq_np,dt=cfg['dt']);mse=float(np.mean((pred.numpy()-truth)**2));energy=float(np.mean(truth**2))
            out.append({'regime':reg,'background_id':f'{reg}-{bi:03d}','action_variant':variant,'physical_code':code,
              'fit_probes':1024,'independent_query_probes':1024,'fit_success':bool(optimized.success),'fit_cost':float(optimized.cost),
              'effect_skill':1-mse/max(energy,1e-30),'query_mse':mse,'true_effect_energy':energy,
              'theta_step_norm':float(np.linalg.norm(optimized.x-theta.numpy())),'privileged':True})
      print(f'Capacity {reg}: {len(split[key]["backgrounds"])} backgrounds',flush=True)
    with (run/'mechanism_capacity_per_background.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    metrics={f'{r}|{v}|U{c+1}':{'mean_skill':float(np.mean([x['effect_skill'] for x in out if x['regime']==r and x['action_variant']==v and x['physical_code']==c])),
      'n_backgrounds':sum(x['regime']==r and x['action_variant']==v and x['physical_code']==c for x in out)}
      for r in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT') for v in ('learned_odd','fixed_tanh_diagnostic') for c in range(4)}
    (run/'mechanism_capacity.json').write_text(json.dumps({'information':'privileged noiseless paired simulator conditional-mean fit; fit inputs and query inputs disjoint; diagnostic only','fit_transitions':1024,'query_transitions':1024,'metrics':metrics},indent=2),encoding='utf-8')
    print('Wrote offline mechanism capacity diagnostic.')
if __name__=='__main__':main()
