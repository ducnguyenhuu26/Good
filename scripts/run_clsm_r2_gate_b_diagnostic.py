"""Diagnostic-only Gate B known-code continuous-strength posterior audit.

Gate A is FAIL. This script runs B only because the user explicitly requested
later diagnostics; it cannot upgrade the sequential A-E research claim.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
import mujoco
from scipy.special import ndtr
from rwt.clsm.config import load_config
from rwt.clsm.inference_dp import cubature_correct
from rwt.clsm.schemas import mujoco_nominal_schema
from experiments.rwt.clsm.physics_wrappers import HiddenLaw,HiddenPhysicsWorld
from scripts import run_clsm_r2_gate_a_diagnostic as ga

GA=ROOT/'results/rwt/clsm_r2_gates/gate_a_diagnostic'
DEFAULT_OUT=ROOT/'results/rwt/clsm_r2_gates/gate_b_diagnostic'
CODES=(1,2,4,3,5,6,7); BUDGETS=(4,8,16,32,64); BOOT=10000
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def grid(code,n):
    active=[j for j in range(3) if code&(1<<j)]
    x,w=np.polynomial.hermite.hermgauss(n);x*=math.sqrt(2);w/=math.sqrt(math.pi)
    xx=np.meshgrid(*([x]*len(active)),indexing='ij');ww=np.meshgrid(*([w]*len(active)),indexing='ij')
    nodes=np.zeros((n**len(active),3));nodes[:,active]=np.stack([a.ravel() for a in xx],1)
    weights=np.prod(np.stack([a.ravel() for a in ww],1),1);return nodes,weights
def phys(x):
    lo=np.array([.5,.5,1.3]);hi=np.array([.85,.85,2.])
    return lo+(hi-lo)*ndtr(x)
def loglike_grid(adapter,obs,act,nxt,code,nodes,chunk=128):
    obs=torch.as_tensor(obs,dtype=torch.float64);act=torch.as_tensor(act,dtype=torch.float64);nxt=torch.as_tensor(nxt,dtype=torch.float64)
    out=np.empty((len(nodes),len(obs)))
    with torch.no_grad():
        for s in range(0,len(nodes),chunk):
            th=torch.as_tensor(nodes[s:s+chunk],dtype=torch.float64);g=len(th);t=len(obs)
            o=obs[None].expand(g,t,-1).reshape(g*t,-1);a=act[None].expand(g,t,-1).reshape(g*t,-1)
            z=torch.full((g*t,),code,dtype=torch.long);xi=th[:,None].expand(g,t,3).reshape(g*t,3)
            mu,var=adapter.model(o,a,z,xi);e=nxt[None].expand(g,t,-1).reshape(g*t,-1)-mu
            lp=-.5*(e.square()/var.clamp_min(1e-8)+var.clamp_min(1e-8).log()+math.log(2*math.pi)).sum(-1)
            out[s:s+g]=lp.reshape(g,t).numpy()
    return np.cumsum(out,axis=1)
def weighted_mean_cov(x,w):
    m=w@x;d=x-m;return m,(d*w[:,None]).T@d
def quant(x,w,q):
    y=np.empty(x.shape[1])
    for j in range(x.shape[1]):
        ii=np.argsort(x[:,j]);cw=np.cumsum(w[ii]);y[j]=x[ii[min(np.searchsorted(cw,q),len(ii)-1)],j]
    return y
def predictive(adapter,probe,code,xis,weights):
    p=len(probe['obs']);dim=adapter.delta_mean.numel();m=np.zeros((p,dim));sec=np.zeros_like(m)
    for s in range(0,len(xis),16):
        x=xis[s:s+16];w=weights[s:s+16]
        o=np.tile(probe['obs'],(len(x),1));a=np.tile(probe['action'],(len(x),1))
        mu,var=ga.predict_numpy(adapter,o,a,np.full(len(o),code),np.repeat(x,p,axis=0),batch=512)
        mu=mu.reshape(len(x),p,dim);var=var.reshape(len(x),p,dim)
        m+=np.einsum('n,npd->pd',w,mu);sec+=np.einsum('n,npd->pd',w,var+mu**2)
    return m,np.maximum(sec-m*m,1e-8)
def score(mu,var,target):
    e=target-mu
    return float(.5*np.mean(e**2/var+np.log(var)+math.log(2*math.pi))),float(np.mean(e**2))
def gaussian_physical(mean,cov,code):
    n,w=grid(code,9);a=[j for j in range(3) if code&(1<<j)]
    chol=np.linalg.cholesky(cov[np.ix_(a,a)]+1e-10*np.eye(len(a)))
    full=np.zeros_like(n);full[:,a]=n[:,a]@chol.T+mean[a]
    p=phys(full);w=w/w.sum();return w@p,p,w
def analytic_fixture():
    x=np.array([-.8,-.2,.1,.5,1.]);y=np.array([-.72,-.13,.24,.41,1.12]);noise=.09
    v=1/(1+np.sum(x*x)/noise);m=v*np.sum(x*y)/noise
    z,w=np.polynomial.legendre.leggauss(4096);g=8*z;w=8*w
    ll=-.5*g*g-.5*np.sum((y[None]-g[:,None]*x[None])**2/noise,axis=1)+np.log(w)
    ll-=ll.max();w=np.exp(ll);w/=w.sum();mn=w@g;vn=w@((g-mn)**2)
    err=max(abs(m-mn),abs(v-vn))
    return {'closed_mean':float(m),'quadrature_mean':float(mn),'closed_variance':float(v),'quadrature_variance':float(vn),'max_abs_error':float(err),'tolerance':1e-8,'passed':bool(err<1e-8)}
def worlds(schema):
    p,h=ga.load_diagnostic_split(GA,'composition_diagnostic')
    ws=[];seen={c:0 for c in CODES}
    for wid in np.unique(p['world_id']):
        q=p['world_id']==wid;i=np.flatnonzero(q)[0]
        code=int(h['code'][i])
        if seen[code]>=12:continue
        seen[code]+=1
        ws.append({'code':code,'id':int(wid),'strength':[float(h[k][i]) for k in ('motor_1','motor_2','damping')],
                   'trajectory':{k:p[k][q] for k in ('obs','action','next_obs')}})
    sp,sh,sm=ga.collect_world_set(ga.ENV_ID,schema,(1,2,4),12,64,51_500,900_000)
    for wid in np.unique(sp['world_id']):
        q=sp['world_id']==wid;i=np.flatnonzero(q)[0];c=int(sh['code'][i])
        ws.append({'code':c,'id':int(wid),'strength':[float(sh[k][i]) for k in ('motor_1','motor_2','damping')],
                   'trajectory':{k:sp[k][q] for k in ('obs','action','next_obs')}})
    bank,bh=ga.load_probe_bank(GA);spec=mujoco.mjtState.mjSTATE_INTEGRATION
    for w in ws:
        c=w['code'];m1,m2,d=w['strength'];law=HiddenLaw(c,m1,m2,d,False)
        env=HiddenPhysicsWorld(ga.ENV_ID,[law]*len(bank['obs']),schema);ys=[]
        try:
            env.reset(seed=123000+w['id'])
            for j in range(len(bank['obs'])):
                y,_,rest=ga.restore_and_step(env,bh['integration_state'][j],bank['action'][j],law,j)
                if not np.allclose(rest,bank['obs'][j],atol=1e-10,rtol=0):raise RuntimeError('probe restore mismatch')
                ys.append(y)
        finally:env.close()
        w['probe']={'obs':bank['obs'].copy(),'action':bank['action'].copy(),'next_obs':np.asarray(ys)}
    return ws
def main_run(out):
    torch.set_num_threads(1);start=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    schema=mujoco_nominal_schema(ga.ENV_ID);config=load_config(ROOT/'experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml')
    ws=worlds(schema);ref=torch.load(ROOT/'results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/checkpoint_reference_r2.pt',map_location='cpu',weights_only=True)
    adapters={}
    for mt in ('oracle_G_z_xi_singletons','oracle_G_z_xi_mixed_seen'):
        adapters[mt]=ga.load_diagnostic_model(GA/'checkpoints'/mt/'seed_4100.pt',mt,schema,config,ref)
    rows=[];coverage=[];conv=[];audit_counts={c:0 for c in CODES}
    for wi,w in enumerate(ws):
        c=w['code'];active=[j for j in range(3) if c&(1<<j)];mt='oracle_G_z_xi_singletons' if c in (1,2,4) else 'oracle_G_z_xi_mixed_seen';ad=adapters[mt]
        # Shared z,xi posterior from one constant-strength world. First 64 common probes are observations.
        n_hi={1:61,2:41,3:15}[len(active)];n_lo={1:41,2:27,3:11}[len(active)]
        nodes,prior=grid(c,n_hi);probe=w['probe'];ll=loglike_grid(ad,probe['obs'][:64],probe['action'][:64],probe['next_obs'][:64],c,nodes)
        audit=audit_counts[c]<3
        if audit:audit_counts[c]+=1
        if audit:
            low,lp=grid(c,n_lo);llo=loglike_grid(ad,probe['obs'][:64],probe['action'][:64],probe['next_obs'][:64],c,low)
        mean=torch.zeros(3,dtype=torch.float64);cov=torch.zeros((3,3),dtype=torch.float64);cov[active,active]=1.
        filt={}
        for t in range(64):
            _,mean,cov,_=cubature_correct(ad.model,torch.as_tensor(probe['obs'][t]),torch.as_tensor(probe['action'][t]),torch.as_tensor(probe['next_obs'][t]),c,mean,cov)
            if t+1 in BUDGETS:filt[t+1]=(mean.clone(),cov.clone())
        strength=np.asarray(w['strength']);true_xi=ga.strength_xi(np.array([c]),np.array([strength[0]]),np.array([strength[1]]),np.array([strength[2]]))[0]
        held={k:v[64:128] for k,v in probe.items()};target=(held['next_obs']-held['obs']-ad.delta_mean.numpy())/ad.delta_scale.numpy()
        for n in BUDGETS:
            lw=np.log(prior)+ll[:,n-1];lw-=lw.max();ww=np.exp(lw);ww/=ww.sum()
            rm,rc=weighted_mean_cov(nodes,ww);pgrid=phys(nodes);rpm=ww@pgrid
            cm,cc=filt[n];cpm,_,_=gaussian_physical(cm.detach().numpy(),cc.detach().numpy(),c)
            rng=np.random.default_rng(99000+wi*97+n);ix=rng.choice(len(ww),128,replace=True,p=ww)
            qm,qv=predictive(ad,held,c,nodes[ix],np.full(128,1/128));qnll,qmse=score(qm,qv,target)
            cx=np.zeros((1,3));cx[0,active]=cm.detach().numpy()[active];cmu,cv=predictive(ad,held,c,cx,np.ones(1));cnll,cmse=score(cmu,cv,target)
            tx=np.zeros((1,3));tx[0,active]=true_xi[active];tm,tv=predictive(ad,held,c,tx,np.ones(1));tnll,tmse=score(tm,tv,target)
            nm,nv=predictive(ad,held,0,np.zeros((1,3)),np.ones(1));nnll,nmse=score(nm,nv,target)
            pcover={}
            for lev,lo,hi in ((50,.25,.75),(80,.1,.9),(90,.05,.95)):
                lower=quant(pgrid,ww,lo);upper=quant(pgrid,ww,hi)
                pcover[str(lev)]={str(j):bool(lower[j]<=strength[j]<=upper[j]) for j in active}
            row={'code':c,'world_id':w['id'],'model':mt,'budget':n,'active':active,
              'xi_reference_mean':rm[active].tolist(),'xi_cubature_mean':cm.detach().numpy()[active].tolist(),
              'xi_mean_abs_error_cubature_vs_reference':float(np.mean(np.abs(cm.detach().numpy()[active]-rm[active]))),
              'xi_covariance_frobenius_error':float(np.linalg.norm(cc.detach().numpy()-rc)),
              'physical_reference_mean':rpm[active].tolist(),'physical_cubature_mean':cpm[active].tolist(),
              'physical_truth':strength[active].tolist(),'physical_reference_abs_error':np.abs(rpm[active]-strength[active]).tolist(),
              'physical_cubature_abs_error':np.abs(cpm[active]-strength[active]).tolist(),'credible_coverage':pcover,
              'reference_predictive_nll':qnll,'reference_probe_mse':qmse,'cubature_predictive_nll':cnll,'cubature_probe_mse':cmse,
              'oracle_plugin_predictive_nll':tnll,'oracle_plugin_probe_mse':tmse,'nominal_predictive_nll':nnll,'nominal_probe_mse':nmse}
            rows.append(row)
            if audit:
                lw2=np.log(lp)+llo[:,n-1];lw2-=lw2.max();w2=np.exp(lw2);w2/=w2.sum();m2=weighted_mean_cov(low,w2)[0]
                conv.append({'code':c,'world_id':w['id'],'budget':n,'coarse_order':n_lo,'fine_order':n_hi,
                    'coarse_points':len(low),'fine_points':len(nodes),'max_abs_xi_mean_diff':float(np.max(np.abs(m2-rm)))})
        acts=np.linalg.norm(w['trajectory']['action'][:64],axis=1)
        vel=np.linalg.norm(w['trajectory']['obs'][:64,schema.joint_velocity_indices],axis=1)
        coverage.append({'code':c,'world_id':w['id'],'mean_action_norm':float(acts.mean()),'action_nonzero_fraction':float(np.mean(acts>1e-6)),'mean_joint_speed':float(vel.mean())})
    summaries=[]
    def boot(v,seed):
        v=np.asarray(v,float);rng=np.random.default_rng(seed);idx=rng.integers(0,len(v),(BOOT,len(v)));z=v[idx].mean(1)
        return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]
    metrics=('xi_mean_abs_error_cubature_vs_reference','xi_covariance_frobenius_error','reference_predictive_nll','reference_probe_mse','cubature_predictive_nll','cubature_probe_mse','oracle_plugin_predictive_nll','oracle_plugin_probe_mse','nominal_predictive_nll','nominal_probe_mse')
    for c in CODES:
      for n in BUDGETS:
       a=[r for r in rows if r['code']==c and r['budget']==n]
       for k in metrics:
        v=[r[k] for r in a];summaries.append({'code':c,'budget':n,'metric':k,'mean':float(np.mean(v)),'world_ci95':boot(v,80000+c*100+n+len(k)),'n_worlds':len(v)})
    max_grid_delta=max((r['max_abs_xi_mean_diff'] for r in conv),default=float('inf'))
    ref_converged=max_grid_delta<=.01 and analytic_fixture()['passed']
    result={'diagnostic_only':True,'sequential_status':'BLOCKED_BY_GATE_A_FAIL',
      'diagnostic_status':'PASS_REFERENCE_NUMERICS' if ref_converged else 'INCONCLUSIVE_REFERENCE_NOT_CONVERGED',
      'scope':'known-code continuous strength; same public decoder for cubature and quadrature; evaluator labels joined after inference',
      'world_count_by_code':{str(c):sum(w['code']==c for w in ws) for c in CODES},
      'budget_definition':'first n of 64 fixed common snapshots adapt; final 64 common snapshots are held out for probe scoring',
      'quadrature':'tensor Gauss-Hermite under xi~N(0,I); 61/41/15 nodes per active axis; lower-order audit on 3 worlds per code',
      'paired_world_bootstrap_draws':BOOT,'analytic_linear_gaussian_fixture':analytic_fixture(),
      'quadrature_mean_convergence_tolerance_xi':.01,'quadrature_max_observed_xi_mean_difference':max_grid_delta,
      'grid_convergence':conv,'action_velocity_coverage':coverage,'world_bootstrap_summaries':summaries,
      'posterior_rows':rows,'calibration_intervals':[{'code':r['code'],'world_id':r['world_id'],'budget':r['budget'],'mechanism':j,
        'nominal_coverage':level,'covered':ok} for r in rows for level,d in r['credible_coverage'].items() for j,ok in d.items()],
      'runtime_seconds':time.perf_counter()-start,
      'limitation':'Gate A failed, so these results isolate known-z inference diagnostics and cannot count as sequential Gate B PASS.'}
    write(out/'strength_reference.json',result)
    with (out/'posterior_vs_quadrature.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    with (out/'physical_multiplier_calibration.csv').open('w',newline='',encoding='utf-8') as f:
        cal=result['calibration_intervals'];wr=csv.DictWriter(f,fieldnames=list(cal[0]));wr.writeheader();wr.writerows(cal)
    write(out/'dataset_manifest.json',{'codes':CODES,'world_counts':result['world_count_by_code'],'public_only_filter_inputs':True,
        'public_test_data_sha256':ga.load_existing_split('test')[2]['public_sha256'],'diagnostic_code_trajectory_source':'Gate A heldout composition diagnostic',
        'common_probe_bank_sha256':ga.digest(GA/'public/matched_probe_bank.npz'),'confirmed_probe_restore_checks':'all passed at 1e-10'})
    (out/'coverage_and_fisher.md').write_text('# Gate B diagnostic coverage and Fisher\n\nGate A failed; this is diagnostic-only. The first 64 transitions of one shared 128-state/action bank are available to the known-z filter; the final 64 common probes are held out. Strength labels are scoring-only. Per-world action/velocity support is in strength_reference.json. Tensor Gauss-Hermite order was checked against a lower order on three preselected worlds per composition; inspect the recorded differences before interpreting reference posteriors.\n',encoding='utf-8')
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);a=p.parse_args()
    x=main_run(a.output_dir.resolve());print(json.dumps({'diagnostic_status':x['diagnostic_status'],'runtime_seconds':x['runtime_seconds'],'worlds_per_code':x['world_count_by_code']},indent=2))
if __name__=='__main__':main()
