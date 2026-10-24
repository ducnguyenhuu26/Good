"""Matched-data Q1 V3 action/operator repair ablations on a frozen fresh split."""
from __future__ import annotations

import csv, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np, torch, yaml
from q1.datasets import rollout,stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS,DynamicsParams,apply_program,response
from q1_v3.mechanisms import LegacyBoundedOddAction,OddActionResponse,mean_delta_v
from q1_v3.operators import MechanismOperatorBank
from q1_v3.program_search import score_program_budgets,predict_effect
from q1_v3.task_inference import fit_task_map
from q1_v3.training import TransitionEpisode,fit_operator_bank,fit_shared_response
from scripts.q1_v3_run import fit_context,tensor_data


def save_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load_params(rows):
    return [DynamicsParams(np.asarray(r['B']),float(r['d']),float(r['b']),float(r['kappa'])) for r in rows]


def old_episodes(params,seeds,cfg,label):
    result=[]
    for i,(p,s) in enumerate(zip(params,seeds)):
        x,y=tensor_data(rollout(p,n=cfg['old_training_transitions'],seed=int(s)+100000,dt=cfg['dt'],sigma=cfg['sigma']))
        result.append(TransitionEpisode(x.detach(),y.detach(),f'{label}-{i:03d}'))
    return result


def restore_checkpoint(path,action_class,seed_for_bank=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if 'action_state' in checkpoint:
        action=action_class().double();action.load_state_dict(checkpoint['action_state']);action.eval()
        return action,checkpoint['prior_mean'].double(),checkpoint['prior_covariance'].double()
    if 'state_dict' in checkpoint:
        bank=MechanismOperatorBank(dtype=torch.float64);bank.load_state_dict(checkpoint['state_dict']);bank.eval()
        return bank,checkpoint.get('opaque_map')
    raise ValueError(f'Unrecognized checkpoint: {path}')


def fit_old_maps(action,prior_mean,prior_cov,episodes,seeds,cfg,seed_offset=0):
    return [fit_context(ep.x,ep.y,action,prior_mean,prior_cov,int(s)+seed_offset,
        {'dt':cfg['dt'],'sigma':cfg['sigma'],'eval_map_starts':1,'map_maxiter':cfg['map_maxiter']})
        for ep,s in zip(episodes,seeds)]


def build_operator_episodes(params,seeds,oldfits,post_cache,programs,physical_to_opaque,label):
    episodes=[]
    for i,(seed,fit) in enumerate(zip(seeds,oldfits)):
        for pi,program in enumerate(programs):
            x,y=post_cache[(i,pi)]
            episodes.append(TransitionEpisode(x,y,f'{label}-{i:03d}',
                tuple(physical_to_opaque[c] for c in program),fit.theta.detach()))
    return episodes


def paired_bootstrap(values,seed,reps=5000):
    v=np.asarray(values,dtype=float)
    if len(v)==0:return [float('nan'),float('nan')]
    draws=np.random.default_rng(seed).choice(v,size=(reps,len(v)),replace=True).mean(1)
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def main():
    out=ROOT/'results/q1/q1_v3_mechanism_repair_dev1';cfg=yaml.safe_load((out/'config.resolved.yaml').read_text(encoding='utf-8'))
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));split=protocol['split_manifest'];seed=int(protocol['seed'])
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.time()
    tr_params=load_params(split['new_train']['backgrounds']);tr_seeds=split['new_train']['seeds']
    in_params=load_params(split['new_inner_train']['backgrounds']);in_seeds=split['new_inner_train']['seeds']
    id_params=load_params(split['development_idlike']['backgrounds']);id_seeds=split['development_idlike']['seeds']
    od_params=load_params(split['development_background_ood']['backgrounds']);od_seeds=split['development_background_ood']['seeds']
    train_old=old_episodes(tr_params,tr_seeds,cfg,'train');inner_old=old_episodes(in_params,in_seeds,cfg,'inner')
    programs=[tuple([i]) for i in range(4)]+[tuple(p) for p in cfg['composition_training_pairs']]
    physical_to_opaque=json.loads((out/'operator_training.json').read_text(encoding='utf-8'))['physical_to_opaque_training_only']
    physical_to_opaque={int(k):int(v) for k,v in physical_to_opaque.items()}
    # Create observed post-stage data exactly once. Every ablation consumes these
    # same tensors and the same opaque labels.
    post_cache={}
    for label,params,seeds in [('train',tr_params,tr_seeds),('inner',in_params,in_seeds)]:
        for bi,(p,s) in enumerate(zip(params,seeds)):
            for pi,program in enumerate(programs):
                x,y=tensor_data(rollout(apply_program(p,program),n=cfg['after_training_transitions'],
                    seed=int(s)+200000+pi*4099,dt=cfg['dt'],sigma=cfg['sigma']))
                post_cache[(label,bi,pi)]=(x.detach(),y.detach())
    train_post={(i,pi):post_cache[('train',i,pi)] for i in range(len(tr_params)) for pi in range(len(programs))}
    inner_post={(i,pi):post_cache[('inner',i,pi)] for i in range(len(in_params)) for pi in range(len(programs))}
    # Preserve the original fitted model exactly for the original-V3 row.
    legacy_action,legacy_mean,legacy_cov=restore_checkpoint(out/'v3_primary.pt',LegacyBoundedOddAction)
    legacy_operator,loaded_map=restore_checkpoint(out/'operator_seed4201_sparse.pt',LegacyBoundedOddAction)
    original_opaque_map={int(k):int(v) for k,v in loaded_map.items()}
    if (out/'action_flexible_odd.pt').exists() and (out/'action_flexible_training.json').exists():
        action,prior_mean,prior_cov=restore_checkpoint(out/'action_flexible_odd.pt',OddActionResponse)
        action_meta=json.loads((out/'action_flexible_training.json').read_text(encoding='utf-8'))
    else:
        action,prior_mean,prior_cov,action_meta=fit_shared_response(train_old,inner_old,seed=seed,
            epochs=cfg['shared_action_epochs'],dt=cfg['dt'],sigma=cfg['sigma'],action_factory=OddActionResponse)
        torch.save({'action_state':action.state_dict(),'prior_mean':prior_mean,'prior_covariance':prior_cov},out/'action_flexible_odd.pt')
        (out/'action_flexible_training.json').write_text(json.dumps(action_meta,indent=2),encoding='utf-8')
    # Old-law MAPs used to supervise operator fitting are fitted separately for
    # each action hypothesis, from the same observed old-stage transitions.
    legacy_train_maps=fit_old_maps(legacy_action,legacy_mean,legacy_cov,train_old,tr_seeds,cfg,700)
    legacy_inner_maps=fit_old_maps(legacy_action,legacy_mean,legacy_cov,inner_old,in_seeds,cfg,700)
    flex_train_maps=fit_old_maps(action,prior_mean,prior_cov,train_old,tr_seeds,cfg,700)
    flex_inner_maps=fit_old_maps(action,prior_mean,prior_cov,inner_old,in_seeds,cfg,700)
    legacy_train=build_operator_episodes(tr_params,tr_seeds,legacy_train_maps,train_post,programs,physical_to_opaque,'train')
    legacy_inner=build_operator_episodes(in_params,in_seeds,legacy_inner_maps,inner_post,programs,physical_to_opaque,'inner')
    flex_train=build_operator_episodes(tr_params,tr_seeds,flex_train_maps,train_post,programs,physical_to_opaque,'train')
    flex_inner=build_operator_episodes(in_params,in_seeds,flex_inner_maps,inner_post,programs,physical_to_opaque,'inner')
    models={}
    # Action-only: flexible action, original unconstrained operator fit.
    # Operator-only: original action, learned selective mechanism assignment.
    # Combined: flexible action plus learned selective assignments.
    audit_path=out/'ablation_training.json'
    prior_audit=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None
    training_log=prior_audit['operator_fits'] if prior_audit else {}
    for name,fit_action,train_eps,inner_eps,selective,sparse,sel_weight in [
        ('action_response_only',action,flex_train,flex_inner,False,cfg['sparsity_penalty'],0.),
        ('operator_selectivity_only',legacy_action,legacy_train,legacy_inner,True,cfg['operator_selectivity']['group_sparsity_weight'],cfg['operator_selectivity']['group_entropy_weight']),
        ('combined_action_and_operator',action,flex_train,flex_inner,True,cfg['operator_selectivity']['group_sparsity_weight'],cfg['operator_selectivity']['group_entropy_weight'])]:
        models[name]={}
        training_log.setdefault(name,{})
        for opseed in cfg['operator_seeds']:
            bank_path=out/f'{name}_operator_seed{opseed}.pt'
            if bank_path.exists() and prior_audit:
                saved=torch.load(bank_path,map_location='cpu',weights_only=False)
                bank=MechanismOperatorBank(dtype=torch.float64,selective=selective);bank.load_state_dict(saved['state_dict']);bank.eval()
                meta=training_log[name][str(opseed)]
            else:
                bank,meta=fit_operator_bank(fit_action,train_eps,inner_eps,seed=opseed,dt=cfg['dt'],sigma=cfg['sigma'],
                    epochs=cfg['operator_epochs'],sparsity=sparse,selective=selective,
                    selectivity_weight=sel_weight)
                torch.save({'state_dict':bank.state_dict(),'opaque_map':physical_to_opaque,'selective':selective},bank_path)
                training_log[name][str(opseed)]=meta
                (out/'ablation_training.json').write_text(json.dumps({'action_response':action_meta,'operator_fits':training_log,
                    'shared_observation_cache':True,'training_programs':[list(p) for p in programs],
                    'transitions_per_background_program':cfg['after_training_transitions'],
                    'physical_to_opaque_fixed_from_original_training_seed':physical_to_opaque},indent=2),encoding='utf-8')
            models[name][opseed]=bank
            print(f'{name} operator seed {opseed} trained; best inner epoch {meta["best_epoch"]}.',flush=True)
    torch.save({'state_dict':legacy_operator.state_dict(),'opaque_map':original_opaque_map},out/'original_v3_operator_copy.pt')
    (out/'ablation_training.json').write_text(json.dumps({'action_response':action_meta,'operator_fits':training_log,
        'shared_observation_cache':True,'training_programs':[list(p) for p in programs],
        'transitions_per_background_program':cfg['after_training_transitions'],
        'physical_to_opaque_fixed_from_original_training_seed':physical_to_opaque},indent=2),encoding='utf-8')
    # Recover original V3 old-task MAPs from the pre-repair evaluation output.
    old_map_rows=list(csv.DictReader((out/'old_map_diagnostics.csv').open(encoding='utf-8')))
    legacy_eval_map={(r['regime'],r['background_id'],tuple(json.loads(r['true_program']))):
        (torch.tensor(json.loads(r['theta'])),torch.tensor(json.loads(r['covariance'])))
        for r in old_map_rows if r['action_variant']=='learned_odd'}
    length_penalty=json.loads((out/'length_penalty_selection.json').read_text(encoding='utf-8'))['selected']
    regimes=[('IDLIKE_DEVELOPMENT',id_params,id_seeds),('BACKGROUND_OOD_DEVELOPMENT',od_params,od_seeds)]
    eval_rows=[];effect_rows=[];map_rows=[]
    operator_candidates=[tuple(physical_to_opaque[i] for i in p) for p in PROGRAMS]
    def eval_one(name,fit_action,prior_m,prior_c,banks,case,theta0,cov0,context,regime,bg_id,program,classes,adapt_x,adapt_y,query_x,query_y,truth_effect=None,code=None):
        for opseed,bank in banks.items():
            budgets=cfg['budgets'] if context=='inferred_old' else [100]
            scoremap=score_program_budgets(adapt_x[:100],adapt_y[:100],theta0,cov0,bank,fit_action,
                operator_candidates,budgets,dt=cfg['dt'],sigma=cfg['sigma'],length_penalty=length_penalty)
            for n,scores in scoremap.items():
                for scoring in ('marginal','point'):
                    key='point_score' if scoring=='point' else 'marginal_score'
                    predrow=min(scores,key=lambda row:row[key]);opaque_pred=tuple(predrow['program'])
                    physical_pred=tuple(next(k for k,v in physical_to_opaque.items() if v==c) for c in opaque_pred)
                    pred_theta=bank.apply_program(theta0,opaque_pred)
                    with torch.no_grad():qpred=mean_delta_v(query_x,pred_theta,fit_action,cfg['dt'])
                    eval_rows.append({'regime':regime,'background_id':bg_id,'true_program':json.dumps(program),'N':n,
                        'ablation':name,'context':context,'optimizer_seed':opseed,'scoring':scoring,
                        'predicted_program':json.dumps(physical_pred),'raw_program_correct':int(physical_pred==program),
                        'functional_class_correct':int(classes[PROGRAMS.index(physical_pred)]==classes[PROGRAMS.index(program)]),
                        'query_delta_v_mse':float((qpred-query_y).square().mean())})
            if code is not None and truth_effect is not None:
                opaque=physical_to_opaque[code]
                predicted=predict_effect(query_x,theta0,bank,opaque,fit_action,dt=cfg['dt']).detach().numpy()
                truth=truth_effect
                mse=float(np.mean((predicted-truth)**2));energy=float(np.mean(truth**2))
                effect_rows.append({'regime':regime,'background_id':bg_id,'optimizer_seed':opseed,
                    'ablation':name,'context':context,'physical_code':code,'effect_skill':1-mse/max(energy,1e-30),
                    'mse':mse,'effect_energy':energy,'predicted_energy':float(np.mean(predicted**2)),'zero_effect_skill':0.0})
    for regime,params_list,seeds in regimes:
        for bi,(p,bseed) in enumerate(zip(params_list,seeds)):
            bg_id=f'{regime}-{bi:03d}';classes=functional_classes(p,dt=cfg['dt'])[0]
            case_rows=[]
            for pi,program in enumerate(PROGRAMS):
                old,adapt,query=stage_pair(p,program,seed=int(bseed)+300000+pi*7919,
                    n_support=cfg['old_evaluation_transitions'],n_query=cfg['adaptation_trajectory'],dt=cfg['dt'],sigma=cfg['sigma'])
                ox,oy=tensor_data(old);ax,ay=tensor_data(adapt);qx,qy=tensor_data(query)
                fitted=fit_context(ox,oy,action,prior_mean,prior_cov,int(bseed)+pi*31,
                    {'dt':cfg['dt'],'sigma':cfg['sigma'],'eval_map_starts':cfg['eval_map_starts'],'map_maxiter':cfg['map_maxiter']})
                map_rows.append({'regime':regime,'background_id':bg_id,'true_program':json.dumps(program),
                    'theta':fitted.theta.tolist(),'covariance':fitted.covariance.tolist(),'converged':fitted.converged,
                    'jacobian_singular_values':fitted.jacobian_singular_values})
                if len(program)==1:
                    true_effect=response(apply_program(p,program),qx.numpy(),dt=cfg['dt'])-response(p,qx.numpy(),dt=cfg['dt'])
                    code=program[0]
                else:true_effect=None;code=None
                case_models=[('action_response_only',action,prior_mean,prior_cov,models['action_response_only']),
                    ('operator_selectivity_only',legacy_action,legacy_mean,legacy_cov,models['operator_selectivity_only']),
                    ('combined_action_and_operator',action,prior_mean,prior_cov,models['combined_action_and_operator'])]
                for name,act,pm,pc,banks in case_models:
                    f=fitted.theta if name in ('action_response_only','combined_action_and_operator') else legacy_eval_map[(regime,bg_id,program)][0]
                    covariance=fitted.covariance if name in ('action_response_only','combined_action_and_operator') else legacy_eval_map[(regime,bg_id,program)][1]
                    if name=='operator_selectivity_only':
                        # Use the frozen original V3 law fit for a clean operator-only ablation.
                        f,covariance=legacy_eval_map[(regime,bg_id,program)]
                    for ctx,theta,cov in [('inferred_old',f,covariance),('population_mean',pm,pc)]:
                        eval_one(name,act,pm,pc,banks,None,theta,cov,ctx,regime,bg_id,program,classes,ax,ay,qx,qy,true_effect,code)
                case_rows.append(pi)
            save_csv(out/f'ablation_progress_{regime}_{bi:02d}.csv',[r for r in eval_rows if r['regime']==regime and r['background_id']==bg_id])
            if bi%4==3:print(f'{regime}: evaluated backgrounds 0-{bi}.',flush=True)
    save_csv(out/'repair_ablation_per_background.csv',eval_rows)
    save_csv(out/'repair_effect_per_background.csv',effect_rows)
    save_csv(out/'repair_old_map_diagnostics.csv',map_rows)
    (out/'ablation_runtime.json').write_text(json.dumps({'seconds':time.time()-start,'evaluation_rows':len(eval_rows),
        'effect_rows':len(effect_rows),'same_fresh_protocol_as_original_v3':True,'same_observed_train_transitions':True,
        'same_evaluation_stage_pair_seeds':True,'seed_disjoint':protocol['all_seed_sets_disjoint']},indent=2),encoding='utf-8')
    print(f'Repair ablations finished: {len(eval_rows)} class rows, {len(effect_rows)} effect rows.',flush=True)


if __name__=='__main__':main()
