"""One complete CPU Q1 V3 mechanism-structured development study."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from q1.datasets import rollout,stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS,DynamicsParams,apply_program,response,sample_params
from scripts.q1_v2_pipeline import collect_backgrounds,gather_forbidden_seeds,rows,sha256
from q1_v3.mechanisms import FixedTanhAction,OddActionResponse,mean_delta_v
from q1_v3.operators import MechanismOperatorBank
from q1_v3.program_search import mean_jacobian,score_program_budgets,predict_effect
from q1_v3.task_inference import fit_task_map
from q1_v3.training import TransitionEpisode,estimate_prior,fit_operator_bank,fit_shared_response

TRAIN_COUNTS={'train':((.75,1.),(.28,.42),(.08,.14),(.18,.28))}

def save_rows(path,items):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)

def tensor_data(tr):
    x,y=rows(tr);return torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64)

def fit_context(x,y,action,mean,cov,seed,cfg):
    return fit_task_map(x,y,action,mean,cov,dt=cfg['dt'],sigma=cfg['sigma'],seed=seed,
        starts=cfg['eval_map_starts'],maxiter=cfg['map_maxiter'])

def cls(params,dt):return functional_classes(params,dt=dt)[0]

def gather_all_background_seeds():
    values,sources=gather_forbidden_seeds()
    for path in sorted((ROOT/'results/q1').rglob('*.json')):
        try:obj=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):continue
        found=set()
        def walk(node):
            if isinstance(node,dict):
                for key,value in node.items():
                    if key=='seeds' and isinstance(value,list):found.update(int(v) for v in value if isinstance(v,(int,float)))
                    walk(value)
            elif isinstance(node,list):
                for value in node:walk(value)
        walk(obj)
        if found-values:
            values.update(found);sources.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),'unique_seed_count':len(found)})
    return values,sources

def score_accuracy(scores,true_program,classes,opaque_to_physical,*,point=False):
    key='point_score' if point else 'marginal_score'
    pred=tuple(opaque_to_physical[c] for c in min(scores,key=lambda r:r[key])['program'])
    return pred,int(classes[PROGRAMS.index(pred)]==classes[PROGRAMS.index(true_program)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/q1_v3.yaml');ap.add_argument('--run-dir',default='results/q1/q1_v3_mechanism_dev1');args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'));out=ROOT/args.run_dir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);started=time.time();seed=int(cfg['seed']);old_cfg=yaml.safe_load((ROOT/'configs/q1_v2_operator_retrain.yaml').read_text(encoding='utf-8'))
    resuming=(out/'protocol.json').exists()
    if resuming:
        protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));seed=int(protocol['seed']);split=protocol['split_manifest']
        def restore(name):
            rows0=split[name]['backgrounds'];return [int(s) for s in split[name]['seeds']],[DynamicsParams(np.asarray(r['B']),float(r['d']),float(r['b']),float(r['kappa'])) for r in rows0]
        tr_seeds,tr_params=restore('new_train');inn_seeds,inn_params=restore('new_inner_train')
        id_seeds,id_params=restore('development_idlike');ood_seeds,ood_params=restore('development_background_ood')
        sources=protocol['collision_audit_sources']
    else:
        forbidden,sources=gather_all_background_seeds();source=seed
        tr_seeds,tr_params=collect_backgrounds(cfg['train_backgrounds'],source,forbidden,'train',TRAIN_COUNTS);forbidden.update(tr_seeds)
        inn_seeds,inn_params=collect_backgrounds(cfg['inner_backgrounds'],source+1,forbidden,'train',TRAIN_COUNTS);forbidden.update(inn_seeds)
        ood_ranges={'dev_ood':{'B_diag':(1.08,1.2),'B_offdiag':(-.12,.12),'d':(.49,.59),'b':(.16,.20),'kappa':(.32,.39)}}
        id_seeds,id_params=collect_backgrounds(cfg['development_idlike_backgrounds'],source+2,forbidden,'train',TRAIN_COUNTS);forbidden.update(id_seeds)
        ood_seeds,ood_params=collect_backgrounds(cfg['development_background_ood_backgrounds'],source+3,forbidden,'dev_ood',ood_ranges);forbidden.update(ood_seeds)
        split={'new_train':{'seeds':tr_seeds,'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in tr_params]},
          'new_inner_train':{'seeds':inn_seeds,'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in inn_params]},
          'development_idlike':{'seeds':id_seeds,'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in id_params]},
          'development_background_ood':{'seeds':ood_seeds,'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in ood_params]},'test_split_used':False}
        protocol={'seed':seed,'branch':__import__('subprocess').check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
          'base_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'device':'cpu','dt':cfg['dt'],'sigma':cfg['sigma'],'protocol_frozen_before_training':True,
          'collision_audit_sources':sources,'split_manifest':split,'all_seed_sets_disjoint':len(set(tr_seeds+inn_seeds+id_seeds+ood_seeds))==sum(map(len,[tr_seeds,inn_seeds,id_seeds,ood_seeds])),
          'programs':[list(p) for p in PROGRAMS],'programs_train':[[i] for i in range(4)]+cfg['composition_training_pairs'],
          'programs_composition_ood':[list(p) for p in PROGRAMS if len(p)==2 and list(p) not in cfg['composition_training_pairs']],
          'exploration':'fixed low-excitation q1.datasets.rollout/stage_pair default action protocol'}
        source_paths=[ROOT/'q1_v3/mechanisms.py',ROOT/'q1_v3/task_inference.py',ROOT/'q1_v3/operators.py',ROOT/'q1_v3/training.py',ROOT/'q1_v3/program_search.py',ROOT/'scripts/q1_v3_run.py',ROOT/args.config]
        protocol['source_sha256_before_training']={p.relative_to(ROOT).as_posix():sha256(p) for p in source_paths}
        protocol['software_before_training']={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__}
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        (out/'split_manifest.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    resolved={**cfg,'config_source':args.config,'branch':protocol['branch'],'base_commit':protocol['base_commit'],'split_manifest':'split_manifest.json','protocol':'protocol.json'}
    (out/'config.resolved.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False),encoding='utf-8')
    if not resuming:(out/'collision_audit.json').write_text(json.dumps({'new_split_seeds_disjoint':protocol['all_seed_sets_disjoint'],'sources':sources,'new_seed_counts':{k:len(v) for k,v in [('train',tr_seeds),('inner',inn_seeds),('idlike',id_seeds),('ood',ood_seeds)]}},indent=2),encoding='utf-8')
    print('Resuming frozen V3 protocol.' if resuming else 'Frozen V3 protocol before fitting.',flush=True)
    def old_data(params,seeds,label):
        eps=[]
        for i,(p,s) in enumerate(zip(params,seeds)):
            tr=rollout(p,n=cfg['old_training_transitions'],seed=s+100000,dt=cfg['dt'],sigma=cfg['sigma'])
            x,y=tensor_data(tr);eps.append(TransitionEpisode(x,y,f'{label}-{i:03d}'))
        return eps
    train_old=old_data(tr_params,tr_seeds,'train');inner_old=old_data(inn_params,inn_seeds,'inner')
    action,prior_mean,prior_cov,action_fit=fit_shared_response(train_old,inner_old,seed=seed,epochs=cfg['shared_action_epochs'],dt=cfg['dt'],sigma=cfg['sigma'])
    torch.save({'state_dict':action.state_dict(),'prior_mean':prior_mean,'prior_covariance':prior_cov,'best_epoch':action_fit['best_epoch']},out/'action_learned_odd.pt')
    (out/'action_training.json').write_text(json.dumps(action_fit,indent=2),encoding='utf-8')
    print(f"Selected action epoch {action_fit['best_epoch']}; creating after-stage observations.",flush=True)
    physical_to_opaque={i:int(x) for i,x in enumerate(np.random.default_rng(seed+404).permutation(4))}
    opaque_to_physical={v:k for k,v in physical_to_opaque.items()}
    pairs=[tuple(p) for p in cfg['composition_training_pairs']];training_programs=[(i,) for i in range(4)]+pairs
    def operator_data(params,seeds,old_eps,label,fit_action,prior_m,prior_c,starts):
        episodes=[];oldfits=[]
        for bi,(p,s,old_ep) in enumerate(zip(params,seeds,old_eps)):
            oldfit=fit_context(old_ep.x,old_ep.y,fit_action,prior_m,prior_c,s+700,{'dt':cfg['dt'],'sigma':cfg['sigma'],'eval_map_starts':starts,'map_maxiter':cfg['map_maxiter']})
            oldfits.append(oldfit)
            for pi,program in enumerate(training_programs):
                post=rollout(apply_program(p,program),n=cfg['after_training_transitions'],seed=s+200000+pi*4099,dt=cfg['dt'],sigma=cfg['sigma'])
                x,y=tensor_data(post);opaque=tuple(physical_to_opaque[c] for c in program)
                episodes.append(TransitionEpisode(x,y,f'{label}-{bi:03d}',opaque,oldfit.theta.detach()))
        return episodes,oldfits
    train_op,train_oldfits=operator_data(tr_params,tr_seeds,train_old,'train',action,prior_mean,prior_cov,1)
    inner_op,inner_oldfits=operator_data(inn_params,inn_seeds,inner_old,'inner',action,prior_mean,prior_cov,1)
    model_banks={};operator_meta={}
    for opseed in cfg['operator_seeds']:
        bank,meta=fit_operator_bank(action,train_op,inner_op,seed=opseed,dt=cfg['dt'],sigma=cfg['sigma'],epochs=cfg['operator_epochs'],sparsity=cfg['sparsity_penalty'])
        nll_bank,nll_meta=fit_operator_bank(action,train_op,inner_op,seed=opseed+8000,dt=cfg['dt'],sigma=cfg['sigma'],epochs=cfg['operator_epochs'],sparsity=0.)
        torch.save({'state_dict':bank.state_dict(),'opaque_map':physical_to_opaque},out/f'operator_seed{opseed}_sparse.pt')
        torch.save({'state_dict':nll_bank.state_dict(),'opaque_map':physical_to_opaque},out/f'operator_seed{opseed}_nll_only.pt')
        model_banks[(opseed,'V3')]=bank;model_banks[(opseed,'V3_NLL_ONLY')]=nll_bank
        operator_meta[str(opseed)]={'sparse':meta,'nll_only':nll_meta}
        print(f'Operators trained for seed {opseed}.',flush=True)
    # Fixed-tanh diagnostic reference: same observations, separately estimated old tasks and operators.
    tanh=FixedTanhAction();tanh_train=[];tanh_inner=[]
    tanh_fit_rows=[]
    for src_eps,ss,tag,params in [(train_old,tr_seeds,'train',tr_params),(inner_old,inn_seeds,'inner',inn_params)]:
        fits0=[fit_context(ep.x,ep.y,tanh,prior_mean,prior_cov,s+9000,{'dt':cfg['dt'],'sigma':cfg['sigma'],'eval_map_starts':1,'map_maxiter':cfg['map_maxiter']}) for ep,s in zip(src_eps,ss)]
        tanh_fit_rows.append((src_eps,ss,tag,params,fits0))
    tanh_prior_mean,tanh_prior_cov=estimate_prior([f.theta.detach() for f in tanh_fit_rows[0][4]])
    for src_eps,ss,tag,params,_ in tanh_fit_rows:
        dest=tanh_train if tag=='train' else tanh_inner
        for i,(ep,p,s) in enumerate(zip(src_eps,params,ss)):
            refit=fit_context(ep.x,ep.y,tanh,tanh_prior_mean,tanh_prior_cov,s+9100,{'dt':cfg['dt'],'sigma':cfg['sigma'],'eval_map_starts':1,'map_maxiter':cfg['map_maxiter']})
            for pi,program in enumerate(training_programs):
                tr=rollout(apply_program(p,program),n=cfg['after_training_transitions'],seed=s+300000+pi*4099,dt=cfg['dt'],sigma=cfg['sigma']);x,y=tensor_data(tr)
                dest.append(TransitionEpisode(x,y,f'{tag}-{i:03d}',tuple(physical_to_opaque[c] for c in program),refit.theta.detach()))
    tanh_bank,tanh_meta=fit_operator_bank(tanh,tanh_train,tanh_inner,seed=cfg['operator_seeds'][0]+16000,dt=cfg['dt'],sigma=cfg['sigma'],epochs=cfg['operator_epochs'],sparsity=cfg['sparsity_penalty'])
    torch.save({'state_dict':tanh_bank.state_dict(),'opaque_map':physical_to_opaque},out/'operator_fixed_tanh_diagnostic.pt')
    action_variants={'learned_odd':action,'fixed_tanh_diagnostic':tanh};variant_priors={'learned_odd':(prior_mean,prior_cov),'fixed_tanh_diagnostic':(tanh_prior_mean,tanh_prior_cov)};variant_banks={'learned_odd':{s:model_banks[(s,'V3')] for s in cfg['operator_seeds']},
       'learned_odd_nll_only':{s:model_banks[(s,'V3_NLL_ONLY')] for s in cfg['operator_seeds']},
       'fixed_tanh_diagnostic':{cfg['operator_seeds'][0]:tanh_bank}}
    operator_meta['fixed_tanh_diagnostic']=tanh_meta
    torch.save({'action_state':action.state_dict(),'prior_mean':prior_mean,'prior_covariance':prior_cov},out/'v3_primary.pt')
    (out/'operator_training.json').write_text(json.dumps({'physical_to_opaque_training_only':physical_to_opaque,'seeds':operator_meta,
      'train_episode_count':len(train_op),'inner_episode_count':len(inner_op),'train_transitions_per_program_background':cfg['after_training_transitions'],
      'training_programs': [list(p) for p in training_programs],'operator_group_matrices':{str(s):{'L':{str(c):variant_banks['learned_odd'][s].L[c].detach().tolist() for c in range(4)},'alpha':{str(c):variant_banks['learned_odd'][s].alpha[c].detach().tolist() for c in range(4)}} for s in cfg['operator_seeds']},
      'prior_mean':prior_mean.tolist(),'prior_covariance':prior_cov.tolist(),'train_old_map_converged':sum(f.converged for f in train_oldfits),'inner_old_map_converged':sum(f.converged for f in inner_oldfits)},indent=2),encoding='utf-8')
    print('Preparing independent development cases.',flush=True)
    # Use a fixed bank seed to select the length penalty using only inner-training episodes.
    length_scores=[]
    for penalty in cfg['length_penalty_grid']:
        values=[]
        for opseed,bank in variant_banks['learned_odd'].items():
            for ep in inner_op:
                scores=score_program_budgets(ep.x[:100],ep.y[:100],ep.old_theta,prior_cov,bank,action,[ep.opaque_program], [100],dt=cfg['dt'],sigma=cfg['sigma'],length_penalty=penalty)[100]
                actual=scores[0] if scores else None
                if actual is not None:values.append(actual['marginal_score'])
        length_scores.append({'penalty':penalty,'mean_inner_known_program_marginal_nll':float(np.mean(values))})
    length_penalty=min(length_scores,key=lambda r:r['mean_inner_known_program_marginal_nll'])['penalty']
    (out/'length_penalty_selection.json').write_text(json.dumps({'inner_train_only':True,'candidates':length_scores,'selected':length_penalty},indent=2),encoding='utf-8')
    regimes={'IDLIKE_DEVELOPMENT':(id_params,id_seeds),'BACKGROUND_OOD_DEVELOPMENT':(ood_params,ood_seeds)}
    cases=[];map_rows=[];action_options=list(action_variants.items())
    for regime,(params_list,seeds) in regimes.items():
      for bi,(p,bseed) in enumerate(zip(params_list,seeds)):
       classes=cls(p,cfg['dt'])
       for pi,program in enumerate(PROGRAMS):
        old,adapt,query=stage_pair(p,program,seed=int(bseed)+300000+pi*7919,n_support=cfg['old_evaluation_transitions'],n_query=cfg['adaptation_trajectory'],dt=cfg['dt'],sigma=cfg['sigma'])
        ox,oy=tensor_data(old);ax,ay=tensor_data(adapt);qx,qy=tensor_data(query)
        fits={}
        for name,a in action_options:
            pm,pc=variant_priors[name]
            fit=fit_context(ox,oy,a,pm,pc,int(bseed)+pi*31,{'dt':cfg['dt'],'sigma':cfg['sigma'],'eval_map_starts':cfg['eval_map_starts'],'map_maxiter':cfg['map_maxiter']})
            fits[name]=fit;map_rows.append({'regime':regime,'background_id':f'{regime}-{bi:03d}','true_program':json.dumps(program),'action_variant':name,
             'converged':fit.converged,'nll':fit.nll,'theta':fit.theta.tolist(),'covariance':fit.covariance.tolist(),'jacobian_singular_values':fit.jacobian_singular_values})
        cases.append({'regime':regime,'background_id':f'{regime}-{bi:03d}','background_seed':int(bseed),'params':p,'true_program':program,'classes':classes,'old_x':ox,'old_y':oy,'adapt_x':ax,'adapt_y':ay,'query_x':qx,'query_y':qy,'fits':fits,'pi':pi,'bi':bi})
      print(f'Generated {regime}: {len(params_list)} backgrounds.',flush=True)
    # Pair each old context with the next background's history for the same true-program trajectory.
    by_case={(c['regime'],c['bi'],c['pi']):c for c in cases}
    eval_rows=[];effect_rows=[];candidate_rows=[];confusions={}
    for ri,case in enumerate(cases):
      regime=case['regime'];classes=case['classes'];program=case['true_program'];bidx=case['bi'];pidx=case['pi']
      nxt=(bidx+1)%(16)
      swap=by_case[(regime,nxt,pidx)]
      true_response=response(apply_program(case['params'],program),case['query_x'].numpy(),dt=cfg['dt'])-response(case['params'],case['query_x'].numpy(),dt=cfg['dt']) if len(program)==1 else None
      for variant,banks in variant_banks.items():
       for opseed,bank in banks.items():
        action_used=action_variants['fixed_tanh_diagnostic'] if variant=='fixed_tanh_diagnostic' else action
        fit=case['fits']['fixed_tanh_diagnostic' if variant=='fixed_tanh_diagnostic' else 'learned_odd']
        var_prior_mean,var_prior_cov=variant_priors['fixed_tanh_diagnostic' if variant=='fixed_tanh_diagnostic' else 'learned_odd']
        context_free=var_prior_mean
        swapped_theta=swap['fits']['fixed_tanh_diagnostic' if variant=='fixed_tanh_diagnostic' else 'learned_odd'].theta
        scoring_contexts=[('inferred_old',fit.theta,fit.covariance)]
        if variant=='learned_odd' and opseed==cfg['operator_seeds'][0]:
          scoring_contexts += [('population_mean',context_free,prior_cov),('swapped_old',swapped_theta,fit.covariance),('identity_operator',fit.theta,fit.covariance)]
        for context_name,theta0,cov0 in scoring_contexts:
          actual_bank=bank if context_name!='identity_operator' else MechanismOperatorBank(dtype=torch.float64)
          # A fresh identity bank is exact identity except for a deliberately tiny random initialization; set it exactly.
          if context_name=='identity_operator':
            with torch.no_grad():
              actual_bank.L.copy_(torch.eye(2,dtype=torch.float64).expand(4,2,2));actual_bank.alpha.zero_()
          candidates=[tuple(physical_to_opaque[i] for i in p) for p in PROGRAMS]
          if context_name=='population_mean':cov0=var_prior_cov
          scoremap=score_program_budgets(case['adapt_x'][:100],case['adapt_y'][:100],theta0,cov0,actual_bank,action_used,candidates,cfg['budgets'],dt=cfg['dt'],sigma=cfg['sigma'],length_penalty=length_penalty)
          for n,scores in scoremap.items():
            for scoring in ('marginal','point'):
              pred,correct=score_accuracy(scores,program,classes,opaque_to_physical,point=(scoring=='point'))
              pred_opaque=tuple(physical_to_opaque[i] for i in pred)
              pred_theta=actual_bank.apply_program(theta0,pred_opaque)
              with torch.no_grad():query_pred=mean_delta_v(case['query_x'],pred_theta,action_used,cfg['dt'])
              query_mse=float((query_pred-case['query_y']).square().mean())
              eval_rows.append({'regime':regime,'background_id':case['background_id'],'background_seed':case['background_seed'],'true_program':json.dumps(program),'N':n,
                'action_variant':variant,'optimizer_seed':opseed,'context':context_name,'scoring':scoring,'predicted_program':json.dumps(pred),
                'raw_program_correct':int(pred==program),'functional_class_correct':correct,'true_class':classes[case['pi']],
                'query_delta_v_mse':query_mse})
            # retain candidate likelihoods for confusion, calibration and audits
            if n==100 and context_name=='inferred_old' and scoring=='marginal':
              confusions.setdefault(f'{regime}|{variant}|{opseed}',[]).append({'true':case['pi'],'pred':PROGRAMS.index(pred)})
        # Oracle code identity effect skill is separate from code search; zero baseline skill is exactly zero.
        if len(program)==1:
          code=program[0];truth=true_response
          for ctx,theta0 in [('inferred_old',fit.theta),('population_mean',prior_mean)]:
            delta=predict_effect(case['query_x'],theta0,bank,physical_to_opaque[code],action_used,dt=cfg['dt']).numpy()
            energy=float(np.mean(truth**2));mse=float(np.mean((delta-truth)**2));effect_rows.append({'regime':regime,'background_id':case['background_id'],
              'background_seed':case['background_seed'],'optimizer_seed':opseed,'action_variant':variant,'context':ctx,'physical_code':code,
              'effect_skill':1-mse/max(energy,1e-30),'mse':mse,'effect_energy':energy,'predicted_energy':float(np.mean(delta**2)),'zero_effect_skill':0.0})
    # Label-level simulator/program oracle is reported only as an offline ceiling.
    oracle_rows=[]
    for c in cases:
      errors=[]
      x=c['adapt_x'].numpy();y=c['adapt_y'].numpy();q=x[:,:2];v=x[:,2:4];acts=x[:,4:]
      for pgram in PROGRAMS:
        law=apply_program(c['params'],pgram);pred=cfg['dt']*(np.tanh(acts@law.B.T)-law.d*v-law.b*np.linalg.norm(v,axis=1,keepdims=True)*v-law.kappa*q)
        errors.append(float(np.mean((pred-y)**2)))
      pred=PROGRAMS[int(np.argmin(errors))];oracle_rows.append({'regime':c['regime'],'background_id':c['background_id'],'true_program':json.dumps(c['true_program']),
        'predicted_program':json.dumps(pred),'functional_class_correct':int(c['classes'][PROGRAMS.index(pred)]==c['classes'][c['pi']])})
    save_rows(out/'per_background.csv',eval_rows);save_rows(out/'effect_per_background.csv',effect_rows);save_rows(out/'old_map_diagnostics.csv',map_rows);save_rows(out/'oracle_program_rows.csv',oracle_rows)
    summaries={}
    for reg in regimes:
      for variant in variant_banks:
       for context_name in ('inferred_old','population_mean','swapped_old','identity_operator'):
        for scoring in ('marginal','point'):
         for n in cfg['budgets']:
          rows0=[r for r in eval_rows if r['regime']==reg and r['action_variant']==variant and r['context']==context_name and r['scoring']==scoring and r['N']==n]
          summaries[f'{reg}|{variant}|{context_name}|{scoring}|N={n}']={'n_rows':len(rows0),'functional_class_accuracy':float(np.mean([r['functional_class_correct'] for r in rows0])),
            'raw_program_accuracy':float(np.mean([r['raw_program_correct'] for r in rows0])),'mean_query_delta_v_mse':float(np.mean([r['query_delta_v_mse'] for r in rows0]))}
    effect_summary={}
    for reg in regimes:
      for variant in variant_banks:
       for ctx in ('inferred_old','population_mean'):
        for code in range(4):
          cell=[r for r in effect_rows if r['regime']==reg and r['action_variant']==variant and r['context']==ctx and r['physical_code']==code]
          per_bg={b:float(np.mean([r['effect_skill'] for r in cell if r['background_id']==b])) for b in sorted({r['background_id'] for r in cell})}
          boot=np.random.default_rng(seed+code).choice(list(per_bg.values()),size=(3000,len(per_bg)),replace=True).mean(1)
          effect_summary[f'{reg}|{variant}|{ctx}|U{code+1}']={'n_backgrounds':len(per_bg),'mean_per_background_skill':float(np.mean(list(per_bg.values()))),
            'background_bootstrap_95pct_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            'ratio_of_mean_mse_to_mean_energy_skill':float(1-np.mean([r['mse'] for r in cell])/max(np.mean([r['effect_energy'] for r in cell]),1e-30)),
            'zero_effect_skill':0.0,'per_background_values':per_bg}
    paired={}
    for reg in regimes:
      for variant in variant_banks:
       for scoring in ('marginal','point'):
        for base in ('population_mean','swapped_old','identity_operator'):
          ea=[r for r in eval_rows if r['regime']==reg and r['action_variant']==variant and r['context']=='inferred_old' and r['scoring']==scoring and r['N']==100]
          bb=[r for r in eval_rows if r['regime']==reg and r['action_variant']==variant and r['context']==base and r['scoring']==scoring and r['N']==100]
          acc=lambda rows0:{b:float(np.mean([r['functional_class_correct'] for r in rows0 if r['background_id']==b])) for b in {r['background_id'] for r in rows0}}
          a=acc(ea);b=acc(bb);common=sorted(set(a)&set(b));diff=[a[k]-b[k] for k in common]
          if not common:continue
          sim=np.random.default_rng(seed).choice(diff,size=(3000,len(diff)),replace=True).mean(1)
          paired[f'{reg}|{variant}|{scoring}|N=100|inferred-minus-{base}']={'mean_difference':float(np.mean(diff)),'background_bootstrap_95pct_ci':[float(np.quantile(sim,.025)),float(np.quantile(sim,.975))],'n_paired_backgrounds':len(common)}
    oracle_summary={reg:float(np.mean([r['functional_class_correct'] for r in oracle_rows if r['regime']==reg])) for reg in regimes}
    classes_report={}
    for reg in regimes:
      for variant in variant_banks:
        rows0=[r for r in eval_rows if r['regime']==reg and r['action_variant']==variant and r['context']=='inferred_old' and r['scoring']=='marginal' and r['N']==100]
        mat=np.zeros((21,21),dtype=int)
        for r in rows0:mat[PROGRAMS.index(tuple(json.loads(r['true_program'])))][PROGRAMS.index(tuple(json.loads(r['predicted_program'])))]+=1
        classes_report[f'{reg}|{variant}']=mat.tolist()
    metrics={'status':'FRESH_V3_DEVELOPMENT_DIAGNOSTIC','no_original_test_split':True,'no_q2':True,
      'marginal_is_linearized_laplace_approximation':True,'metrics':summaries,'effect_skill_by_code':effect_summary,
      'paired_baselines':paired,'oracle_program_functional_class_accuracy':oracle_summary,'raw_program_confusion_matrices':classes_report,
      'composition_ood_pairs':[list(p) for p in PROGRAMS if len(p)==2 and list(p) not in cfg['composition_training_pairs']],
      'permitted_claim':'grey-box system identification with a disclosed physics-informed mechanism class; not unsupervised mechanism discovery'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (out/'effect_skill_by_code.json').write_text(json.dumps(effect_summary,indent=2),encoding='utf-8')
    (out/'paired_baselines.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    (out/'old_map_diagnostics.json').write_text(json.dumps({'rows':len(map_rows),'mean_converged':float(np.mean([r['converged'] for r in map_rows])),'rows_by_action':{name:sum(r['action_variant']==name for r in map_rows) for name,_ in action_options}},indent=2),encoding='utf-8')
    action_curve=action_fit['curve'];operator_meta['training_runtime_seconds']=time.time()-started
    (out/'training_audit.json').write_text(json.dumps({'action_training':action_fit,'operator_training':operator_meta,'full_coverage':True,
      'uniform_opaque_code_mapping_fixed_before_training':physical_to_opaque,'train_after_episodes_per_seed':len(train_op),'inner_after_episodes':len(inner_op)},indent=2),encoding='utf-8')
    (out/'training_curves.json').write_text(json.dumps({'action':action_curve,'operators':operator_meta},indent=2),encoding='utf-8')
    # Diagnostics-only: matched, noise-free true-parameter mechanism representation scores.
    capacity=[]
    for reg,(plist,seeds) in regimes.items():
      for bi,(p,s) in enumerate(zip(plist,seeds)):
        x=np.random.default_rng(int(s)+773).uniform(-.7,.7,(1024,6));x[:,2:4]*=.7
        xt=torch.as_tensor(x,dtype=torch.float64)
        for name,a in action_options:
          for code in range(4):
            after=apply_program(p,(code,));yb=response(after,x,dt=cfg['dt'])-response(p,x,dt=cfg['dt'])
            y=torch.as_tensor(yb,dtype=torch.float64);theta=torch.tensor([*p.B.reshape(-1),np.log(p.d),np.log(p.b),np.log(p.kappa)],dtype=torch.float64)
            # Local least-squares parameter displacement at the true baseline is a privileged ceiling only.
            J=mean_jacobian(xt,theta,a,cfg['dt']);delta=torch.linalg.lstsq(J,y.reshape(-1)).solution
            pred=J@delta;skill=1-float((pred-y.reshape(-1)).square().mean())/max(float(y.square().mean()),1e-30)
            capacity.append({'regime':reg,'background_id':f'{reg}-{bi:03d}','action_variant':name,'physical_code':code,'effect_skill':skill,'fit_probes':1024,'query_probes':1024,'privileged':True})
    save_rows(out/'mechanism_capacity_per_background.csv',capacity)
    (out/'mechanism_capacity.json').write_text(json.dumps({'type':'offline privileged local matched-input linearized mechanism ceiling; simulator response labels excluded from learner','mean_skill_by_regime_variant_code':{f'{reg}|{variant}|U{code+1}':float(np.mean([r['effect_skill'] for r in capacity if r['regime']==reg and r['action_variant']==variant and r['physical_code']==code])) for reg in regimes for variant,_ in action_options for code in range(4)},'n_rows':len(capacity)},indent=2),encoding='utf-8')
    (out/'source_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha256(p) for p in [ROOT/'q1_v3/mechanisms.py',ROOT/'q1_v3/task_inference.py',ROOT/'q1_v3/operators.py',ROOT/'q1_v3/training.py',ROOT/'q1_v3/program_search.py',ROOT/'scripts/q1_v3_run.py',ROOT/args.config]},indent=2),encoding='utf-8')
    print(f'V3 completed: {len(eval_rows)} evaluation rows in {time.time()-started:.1f}s. Outputs: {out}',flush=True)

if __name__=='__main__':main()
