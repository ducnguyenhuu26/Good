"""Independent grid-vs-R2-factorized filter stress test for Gate C.

Runs diagnostic-only because Gates A and B did not pass. The finite xi grid is
an independent exact reference for the declared discretized two-law system.
"""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve1d

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rwt/clsm_r2_gates/gate_c_diagnostic'
GRID=np.linspace(-4,4,201); DX=GRID[1]-GRID[0]; HORIZON=40; T=HORIZON+1
KAPPAS=(6.,20.); SEPARATIONS=('weak','strong'); DRIFTS=(.005,.1); RESETS=(0.,.15); SEEDS=range(6)

def normpdf(x,m,v): return np.exp(-.5*(x-m)**2/v)/math.sqrt(2*math.pi*v)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def prior_grid():
    p=normpdf(GRID,0,1);return p/p.sum()
PRIOR=prior_grid()
def hazard(code,age,kappa,shape=1.4):
    return -math.expm1(-(((age+1)/kappa)**shape-(age/kappa)**shape))
def drift_kernel(q):
    sig=math.sqrt(q);radius=max(1,int(math.ceil(5*sig/DX)))
    x=np.arange(-radius,radius+1)*DX;k=normpdf(x,0,q);return k/k.sum()
def emission(code,xi,separation,noise):
    gap=.10 if separation=='weak' else .75
    bias=(0.,gap)[code]
    gain=(1.,.86 if separation=='weak' else .52)[code]
    curvature=.10 if separation=='weak' else .18
    return bias+gain*xi+curvature*xi*xi
def exact_step(p,y,params,log_score):
    kappa,q,rho,sep,noise=params; likelihood=np.stack([normpdf(y,emission(z,GRID,sep,noise),noise**2) for z in range(2)])
    un=p*likelihood[:,None,:]; evidence=un.sum(); log_score+=math.log(max(evidence,1e-300)); f=un/max(evidence,1e-300)
    nextp=np.zeros_like(p); kernel=drift_kernel(q)
    for z in range(2):
      for age in range(T):
        mass=f[z,age]; total=mass.sum()
        if total<1e-300:continue
        h=hazard(z,age,kappa); st=(1-h)*(1-rho); re=(1-h)*rho; sw=h
        if age+1<T:
            moved=convolve1d(mass,kernel,mode='constant',cval=0.)
            if moved.sum()>0:moved*=total/moved.sum()
            nextp[z,age+1]+=st*moved+re*total*PRIOR
        nextp[1-z,0]+=sw*total*PRIOR
    nextp/=max(nextp.sum(),1e-300)
    return f,nextp,log_score
def cubature_update(m,v,y,z,sep,noise):
    sd=math.sqrt(max(v,1e-12));x=np.array([m-sd,m+sd]);w=np.array([.5,.5])
    fy=emission(z,x,sep,noise);my=float(w@fy);vy=float(w@((fy-my)**2)+noise**2)
    cross=float(w@((x-m)*(fy-my)));gain=cross/vy
    return my,vy,m+gain*(y-my),max(v-gain*cross,1e-9)
def approx_step(p,m,v,y,params,log_score):
    kappa,q,rho,sep,noise=params
    ev=[];nm=[];nv=[]
    for z in range(2):
        my,vy,mu,cv=cubature_update(m[z],v[z],y,z,sep,noise)
        ev.append(float(normpdf(y,my,vy)));nm.append(mu);nv.append(cv)
    evidence=sum(p[z].sum()*ev[z] for z in range(2));log_score+=math.log(max(evidence,1e-300))
    f=p.copy()
    for z in range(2):f[z]*=ev[z]/max(evidence,1e-300)
    means=np.asarray(nm);vars=np.asarray(nv)
    filtered_means=means.copy();filtered_vars=vars.copy()
    nextp=np.zeros_like(p);incoming=[[] for _ in range(2)]
    for z in range(2):
      for age in range(T):
        mass=float(f[z,age]);h=hazard(z,age,kappa);st=(1-h)*(1-rho);re=(1-h)*rho;sw=h
        if age+1<T:
            nextp[z,age+1]+=mass*(st+re)
            incoming[z].append((mass*st,means[z],vars[z]+q))
            incoming[z].append((mass*re,0.,1.))
        nextp[1-z,0]+=mass*sw;incoming[1-z].append((mass*sw,0.,1.))
    nextp/=max(nextp.sum(),1e-300)
    for z in range(2):
        total=sum(a for a,_,_ in incoming[z])
        if total<=1e-300:continue
        mm=sum(a*b for a,b,_ in incoming[z])/total
        vv=sum(a*(c+(b-mm)**2) for a,b,c in incoming[z])/total
        means[z],vars[z]=mm,max(vv,1e-9)
    return f,nextp,means,vars,filtered_means,filtered_vars,log_score
def simulate(params,rng):
    kappa,q,rho,sep,noise=params;z=int(rng.integers(2));age=0;xi=float(rng.normal());ys=[];truth=[]
    for t in range(HORIZON):
        y=float(emission(z,xi,sep,noise)+rng.normal(0,noise));ys.append(y);truth.append((z,age,xi))
        h=hazard(z,age,kappa)
        u=rng.random()
        if u<h:z=1-z;age=0;xi=float(rng.normal())
        elif u<h+(1-h)*rho:age+=1;xi=float(rng.normal())
        else:age+=1;xi+=float(rng.normal(0,math.sqrt(q)))
    return np.asarray(ys),truth
def switch_probability(p,kappa):
    return float(sum(p[z,a].sum()*hazard(z,a,kappa) for z in range(2) for a in range(T)))
def one_history(true_params,approx_params,seed,case):
    rng=np.random.default_rng(seed);ys,truth=simulate(true_params,rng)
    p=np.zeros((2,T,len(GRID)));p[:,0,:]=.5*PRIOR
    ap=np.zeros((2,T));ap[:,0]=.5;mean=np.zeros(2);var=np.ones(2)
    score_e=score_a=0.;rows=[];age_sep=[]
    for t,y in enumerate(ys):
        prev_e,prev_a=score_e,score_a
        f,p,score_e=exact_step(p,y,true_params,score_e)
        af,ap,mean,var,filtered_mean,filtered_var,score_a=approx_step(ap,mean,var,y,approx_params,score_a)
        # Marginalize strength but retain the complete joint code-age state.
        exact_zu=f.sum(-1);approx_zu=af
        tv_joint=.5*float(np.abs(exact_zu-approx_zu).sum())
        sw_e=switch_probability(f,true_params[0]);sw_a=switch_probability(af,approx_params[0])
        conditional=[];conditional_weights=[]
        for z in range(2):
            for a in range(T):
                w=f[z,a]
                if w.sum()>.005:
                    ex=float(w@GRID/w.sum());conditional.append(abs(ex-filtered_mean[z]));conditional_weights.append(float(w.sum()))
                    if t>10:age_sep.append(ex)
        exact_xi=f.sum(axis=(0,1));exact_xi/=max(exact_xi.sum(),1e-300)
        cdf=np.cumsum(exact_xi)
        exact_lo=float(GRID[min(np.searchsorted(cdf,.05),len(GRID)-1)])
        exact_hi=float(GRID[min(np.searchsorted(cdf,.95),len(GRID)-1)])
        code_w=af.sum(axis=1);code_w/=max(code_w.sum(),1e-300)
        rg=np.random.default_rng(seed*10000+t)
        sampled_z=rg.choice(2,size=4096,p=code_w)
        approx_xi=rg.normal(filtered_mean[sampled_z],np.sqrt(np.maximum(filtered_var[sampled_z],1e-9)))
        approx_lo,approx_hi=np.quantile(approx_xi,[.05,.95])
        rows.append({'case':case,'seed':seed,'time':t+1,'joint_code_age_tv':tv_joint,
          'next_switch_abs_error':abs(sw_e-sw_a),'exact_step_log_score':score_e-prev_e,
          'approx_step_log_score':score_a-prev_a,
          'xi_age_conditional_mean_error_massweighted':float(np.average(conditional,weights=conditional_weights)) if conditional else 0.,
          'exact_xi_90pct_coverage':float(exact_lo<=truth[t][2]<=exact_hi),
          'approx_xi_90pct_coverage':float(approx_lo<=truth[t][2]<=approx_hi),
          'exact_switch_probability':sw_e,'approx_switch_probability':sw_a,
          'approx_prequential_log_score':score_a,'exact_prequential_log_score':score_e})
    # Counterexample diagnostic: same code, different ages yield distinct grid posterior means.
    last=rows[-1]
    exact_state_means={}
    for z in range(2):
        vals=[]
        for a in range(T):
            w=p[z,a]
            if w.sum()>.01:vals.append(float(w@GRID/w.sum()))
        if len(vals)>1:exact_state_means[str(z)]=max(vals)-min(vals)
    return rows,{'case':case,'seed':seed,'mean_joint_tv':float(np.mean([x['joint_code_age_tv'] for x in rows])),
       'max_joint_tv':float(max(x['joint_code_age_tv'] for x in rows)),
       'mean_switch_abs_error':float(np.mean([x['next_switch_abs_error'] for x in rows])),
       'prequential_logscore_difference_approx_minus_grid':score_a-score_e,
       'same_code_age_conditional_xi_range':exact_state_means}
def main_run(out):
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();trace=[];summaries=[];case_id=0
    # Freeze diagnostic parameter grid before this run; every factor has low/high levels.
    for kappa in KAPPAS:
      for sep in SEPARATIONS:
       for q in DRIFTS:
        for rho in RESETS:
         case_id+=1;case=f'k{kappa:g}_{sep}_q{q:g}_r{rho:g}'
         true=(kappa,q,rho,sep,.65 if sep=='weak' else .24)
         approx=true if case_id%2 else (kappa*1.35,q*1.8,rho,sep,true[-1])
         reps=[]
         for seed in SEEDS:
            rows,rec=one_history(true,approx,seed+case_id*100,case);trace.extend(rows);reps.append(rec)
         summaries.extend(reps)
    # Sequential exact categorical regression suite is separately run by pytest.
    with (out/'event_marginals.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(trace[0]));wr.writeheader();wr.writerows(trace)
    with (out/'approximation_error_curves.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(summaries[0]));wr.writeheader();wr.writerows(summaries)
    decision={'diagnostic_only':True,'sequential_status':'BLOCKED_BY_GATE_A_FAIL_AND_GATE_B_INCONCLUSIVE',
      'grid_reference':'two-code age-strength exact forward recursion on xi grid of 201 points, dx=.04; drift convolution row-normalized',
      'factors':{'duration_scale':list(KAPPAS),'emission_separation':list(SEPARATIONS),'drift_variance':list(DRIFTS),'reset_probability':list(RESETS),'seeds_per_cell':len(SEEDS)},
      'cells':case_id,'history_count':len(summaries),'max_observation_steps':HORIZON,
      'exact_categorical_tests':'tests/clsm/test_inference_dp.py and tests/clsm/test_age_audit.py; result in categorical_reference.json',
      'continuous_joint_filter_summaries':summaries,'runtime_seconds':time.perf_counter()-start,
      'limitation':'Synthetic diagnostic does not validate HalfCheetah emission adequacy or learned evolution; finite xi grid has stated discretization and boundary truncation.'}
    write(out/'joint_grid_reference.json',decision)
    write(out/'categorical_reference.json',{'max_tv_tolerance':1e-10,'reference':'Independent exact probability recursion and event-count regression in tests/clsm/test_inference_dp.py and tests/clsm/test_age_audit.py','status':'PENDING_PYTEST'})
    return decision
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args()
    r=main_run(a.output_dir.resolve());print(json.dumps({'diagnostic_only':r['diagnostic_only'],'cells':r['cells'],'histories':r['history_count'],'runtime_seconds':r['runtime_seconds']},indent=2))
if __name__=='__main__':main()
