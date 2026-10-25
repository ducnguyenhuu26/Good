import time,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scipy.optimize import least_squares
from q1.datasets import rollout,stage_pair
from q1.simulator import sample_params,apply_program,response as sim_response, PROGRAMS
from q1_v4.stage_contrast import learn_opaque_codes
from q1_v5.model import ProgramModel,fit_theta,response,DT,SIGMA,joint_map,normal_mixture_shrink
from q1_v5.joint_refit import joint_refit

def xy(t):return np.c_[t.states[:-1],t.actions],t.delta_v

def train_shared(train):
    """Joint fit psi(z) with one theta per training background, no simulator parameters."""
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    x=torch.tensor(np.stack([t[0] for t in train]),dtype=torch.float64)
    y=torch.tensor(np.stack([t[1] for t in train]),dtype=torch.float64)
    initial=np.r_[np.eye(2).reshape(-1)*.9,np.log([.35,.12,.23])]
    theta=torch.nn.Parameter(torch.tensor(np.tile(initial,(len(train),1)),dtype=torch.float64))
    # 3-unit positive-weight saturating mixture; odd, monotone, bounded, unit slope at zero.
    raw_weights=torch.nn.Parameter(torch.tensor([-.5,1.0,-.5],dtype=torch.float64))
    raw_slopes=torch.nn.Parameter(torch.tensor([-1.25,-.4,1.0],dtype=torch.float64))
    opt=torch.optim.LBFGS([theta,raw_weights,raw_slopes],lr=.55,max_iter=360,tolerance_grad=1e-11,tolerance_change=1e-12,line_search_fn='strong_wolfe',history_size=25)
    iterations=[0]
    def coeffs():
        weights=torch.softmax(raw_weights,dim=0)
        slopes=.25+2.25*torch.sigmoid(raw_slopes)
        return slopes,weights
    def closure():
        opt.zero_grad();slopes,weights=coeffs();B=theta[:,:4].reshape(-1,2,2)
        z=torch.einsum('bij,bnj->bni',B,x[:,:,4:6])
        psi=(weights*torch.tanh(z[...,None]*slopes)).sum(-1)/(weights*slopes).sum()
        rates=theta[:,4:].exp();v=x[:,:,2:4];q=x[:,:,:2];speed=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        pred=.05*(psi-rates[:,0,None,None]*v-rates[:,1,None,None]*speed*v-rates[:,2,None,None]*q)
        loss=.5*((pred-y)/SIGMA).square().mean()+.00001*(weights.square().sum())
        loss.backward();iterations[0]+=1
        return loss
    opt.step(closure)
    val=float(closure().detach())
    with torch.no_grad():
        slopes,weights=coeffs()
        return {'slopes':slopes.detach().numpy().tolist(),'weights':weights.detach().numpy().tolist()},theta.detach().numpy(),val

def run():
    import argparse
    ap=argparse.ArgumentParser(description='Observed-data CPU-only Q1 V5 development diagnostic')
    ap.add_argument('--seed',type=int,default=20261024444)
    ap.add_argument('--train-backgrounds',type=int,default=40)
    ap.add_argument('--dev-backgrounds',type=int,default=16)
    ap.add_argument('--output',type=str,default='results/q1/q1_v5_semiparametric_dev1')
    args=ap.parse_args()
    out=ROOT/args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'will not overwrite: {out}')
    out.mkdir(parents=True,exist_ok=True)
    start=time.time();trainseed=args.seed
    trainbg=args.train_backgrounds;devbg=args.dev_backgrounds
    rng=np.random.default_rng(trainseed); physical_to_opaque={i:int(j) for i,j in enumerate(rng.permutation(4))}
    pairs={v:[] for v in physical_to_opaque.values()};before=[];after={i:[] for i in range(4)}
    train_params=[]
    for bi in range(trainbg):
        p=sample_params(np.random.default_rng(trainseed+bi*97+33),'train');train_params.append(p)
        old=xy(rollout(p,n=512,seed=trainseed+bi*811+191,sigma=SIGMA));before.append(old)
        for physical in range(4):
            newp=apply_program(p,(physical,))
            new=xy(rollout(newp,n=512,seed=trainseed+bi*811+90119+physical*123,sigma=SIGMA))
            pairs[physical_to_opaque[physical]].append((*old,*new));after[physical].append(new)
    c,thetas,loss=train_shared(before)
    print('TRAIN SHARED',c,'train loss=',loss,'elapsed',time.time()-start,flush=True)
    learned=learn_opaque_codes(pairs,max_degree=5)
    print('MECHANISM ASSIGNMENTS',[(k,p.mechanism,p.ratio) for k,p in learned.items()],flush=True)
    # Stage-one nonlinear learned-theta fit on per-code after transitions.
    full_after={i:[] for i in range(4)}
    for physical in range(4):
        for bi,(x,y) in enumerate(after[physical]):
            t,_,_=fit_theta(x,y,c,guess=thetas[bi]);full_after[physical].append(t)
    ops={}
    for physical,opaque in physical_to_opaque.items():
        mech=learned[opaque].mechanism
        if mech=='action':
            B0=np.stack([t[:4].reshape(2,2) for t in thetas]);B1=np.stack([t[:4].reshape(2,2) for t in full_after[physical]])
            matrix=np.concatenate([B0[i] for i in range(trainbg)],axis=1)
            target=np.concatenate([B1[i] for i in range(trainbg)],axis=1)
            L=target@np.linalg.pinv(matrix)
            ops[opaque]={'mechanism':'action','L':L}
            print('ACTION OP',opaque,L,'matrix error',float(np.linalg.norm(target-L@matrix)/np.linalg.norm(target)),flush=True)
        else:
            j=['linear_drag','quadratic_drag','position_feedback'].index(mech)
            ratios=np.array([np.exp(full_after[physical][bi][4+j]-thetas[bi][4+j]) for bi in range(trainbg)])
            ratio=float(np.median(ratios))
            ops[opaque]={'mechanism':mech,'ratio':ratio,'index':j}
            print('RATE OP',opaque,mech,ratio,'stagecontrast',learned[opaque].ratio,flush=True)
    ops,thetas,joint_steps=joint_refit(c,before,after,thetas,ops,physical_to_opaque,iterations=210)
    print('JOINT REFIT updates',joint_steps,'ops',[(k,v.get('ratio')) for k,v in ops.items()],flush=True)
    # Robust train-only empirical prior; avoids OOD over-shrinkage.
    mu=np.mean(thetas,axis=0);sd=np.maximum(np.std(thetas,axis=0,ddof=1)*2,[.15,.15,.15,.15,.3,.5,.3])
    model=ProgramModel(c,ops,mu,sd)
    for physical in range(4):
        code=physical_to_opaque[physical]
        print('operator',physical,'opaque',code,'type',ops[code]['mechanism'])
    print('TRAINING TIME',time.time()-start,flush=True)
    budgets=[5,10,20,50,100]
    programs=[tuple(physical_to_opaque[i] for i in pr) for pr in PROGRAMS]
    metrics={};rows=[];effects=[];max_dev=devbg
    mu_b=float(np.mean(np.exp(thetas[:,5])));sd_b=max(float(np.std(np.exp(thetas[:,5]),ddof=1)),.01)
    for regime,region in [('IDLIKE_DEVELOPMENT','train'),('BACKGROUND_OOD_DEVELOPMENT','test_background_ood')]:
        for bi in range(max_dev):
            bg_seed=trainseed+9_000_000+(1 if regime.startswith('BACKGROUND_') else 0)*3_000_000+bi*1987
            p=sample_params(np.random.default_rng(bg_seed),region)
            for pi,actual in enumerate(PROGRAMS):
                old,adapt,query=stage_pair(p,actual,seed=bg_seed+pi*7477+99,n_support=96,n_query=256,sigma=SIGMA)
                x0,y0=xy(old);xa,ya=xy(adapt);xq,yq=xy(query)
                theta,cov,mse=model.old_posterior(x0,y0)
                scores=model.score_candidates(xa[:100],ya[:100],theta,cov,programs,budgets)
                if len(actual)==1:
                    code=physical_to_opaque[actual[0]]
                    if actual[0] in (2,3):
                        changed=model.transform(theta,(code,));prior_changed=model.transform(mu,(code,))
                        actual_change=sim_response(apply_program(p,actual),xq)-sim_response(p,xq)
                        delta=response(xq,changed,c)-response(xq,theta,c)
                        prior_delta=response(xq,prior_changed,c)-response(xq,mu,c)
                        energy=float(np.mean(actual_change**2))
                        candidate=(code,)
                        jm,jcov,jloss=joint_map(model,x0,y0,xa[:100],ya[:100],candidate,initial_theta=theta)
                        delta_joint=response(xq,model.transform(jm,candidate),c)-response(xq,jm,c)
                        old_b=np.exp(theta[5]);old_se=old_b*np.sqrt(max(cov[5,5],1e-30))
                        jm_b=np.exp(jm[5]);jm_se=jm_b*np.sqrt(max(jcov[5,5],1e-30))
                        shrink_old,old_resp=normal_mixture_shrink(old_b,old_se,mu_b,sd_b)
                        shrink_joint,jm_resp=normal_mixture_shrink(jm_b,jm_se,mu_b,sd_b)
                        ebtheta=theta.copy();ebtheta[5]=np.log(shrink_old)
                        ebjoint=jm.copy();ebjoint[5]=np.log(shrink_joint)
                        delta_old_eb=response(xq,model.transform(ebtheta,candidate),c)-response(xq,ebtheta,c)
                        delta_joint_eb=response(xq,model.transform(ebjoint,candidate),c)-response(xq,ebjoint,c)
                        effects.append({'regime':regime,'bg':bi,'U':actual[0]+1,
                            'skill_old':1-float(np.mean((delta-actual_change)**2))/energy,
                            'skill_prior':1-float(np.mean((prior_delta-actual_change)**2))/energy,
                            'skill_joint':1-float(np.mean((delta_joint-actual_change)**2))/energy,
                            'skill_old_eb':1-float(np.mean((delta_old_eb-actual_change)**2))/energy,
                            'skill_joint_eb':1-float(np.mean((delta_joint_eb-actual_change)**2))/energy})
                for n in budgets:
                    score,pr,_=scores[n][0]; predicted=next(PROGRAMS[i] for i,codepr in enumerate(programs) if codepr==pr)
                    pred_theta=model.transform(theta,pr)
                    mse=float(np.mean((response(xq,pred_theta,c)-yq)**2))
                    rows.append({'regime':regime,'bg':bi,'true_program':list(actual),'pred_program':list(predicted),'N':n,'correct':int(predicted==actual),'mse':mse})
        print('REGIME FINISHED',regime,'elapsed',time.time()-start,flush=True)
    for reg in ['IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT']:
        for n in budgets:
            select=[z for z in rows if z['regime']==reg and z['N']==n]
            metrics[f'{reg}|N={n}']={'class_accuracy_unadjusted':float(np.mean([t['correct'] for t in select])),'mean_query_mse':float(np.mean([t['mse'] for t in select]))}
    import csv
    with (out/'effect_rows.csv').open('w',newline='') as ef:
        ew=csv.DictWriter(ef,fieldnames=effects[0].keys());ew.writeheader();ew.writerows(effects)
    for reg in ['IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT']:
        for u in [3,4]:
            rr=[z for z in effects if z['regime']==reg and z['U']==u]
            metrics[f'{reg}|U{u}']={'mean_effect_skill':float(np.mean([z['skill_old'] for z in rr])),'population_baseline':float(np.mean([z['skill_prior'] for z in rr])),'skill_joint':float(np.mean([z['skill_joint'] for z in rr])),'skill_old_eb':float(np.mean([z['skill_old_eb'] for z in rr])),'skill_joint_eb':float(np.mean([z['skill_joint_eb'] for z in rr]))}
    def canonical(p):
        p=list(p)
        return tuple(p) if len(p)==2 and set(p)=={0,1} else tuple(sorted(p))
    for reg in ['IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT']:
        for n in budgets:
            rr=[r for r in rows if r['regime']==reg and r['N']==n]
            metrics[f'{reg}|N={n}']['functional_class_accuracy']=float(np.mean([
                canonical(r['true_program'])==canonical(r['pred_program']) for r in rr]))
    obj={'method'  :'learned_shared_odd_polynomial_operators_laplace','train_backgrounds':trainbg,'development_backgrounds':devbg,'train_loss':loss,'c':c,'operator_types':{str(k):{'mechanism':v['mechanism'],'L':v['L'].tolist() if 'L' in v else None,'ratio':v.get('ratio')} for k,v in ops.items()},'metrics':metrics,'elapsed':time.time()-start,'rows':len(rows),'joint_refit_iterations':joint_steps,'protected_test_used':False,'status':'EXPLORATORY_DEV_ONLY_NOT_Q1_PASS','seed':args.seed}
    (out/'metrics.json').write_text(json.dumps(obj,indent=2))
    import csv
    with (out/'per_program.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('FINAL',json.dumps(metrics,indent=2),'elapsed',time.time()-start,flush=True)

if __name__=='__main__':run()
