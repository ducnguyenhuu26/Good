"""One bounded Q1 V2 operator repair cycle on fresh development backgrounds."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from q1.datasets import rollout,stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS,DynamicsParams,apply_program,response,sample_params
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import GaussianPosterior,gaussian_posterior
from q1_v2.inference import composed,rank_programs
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import (ObservedEffectTask,ObservedEpisode,design,evaluate_basis_nll,
    evaluate_effect_tasks,fit_basis,fit_operator_bank,fit_operator_bank_effect_aware)
from scripts.q1_v2_pipeline import (collect_backgrounds,gather_forbidden_seeds,make_training_data,rows)

TRAIN_COUNTS={'train':((.75,1.),(.28,.42),(.08,.14),(.18,.28))}
OOD_RANGES={'dev_ood':{'B_diag':(1.08,1.2),'B_offdiag':(-.12,.12),'d':(.49,.59),'b':(.16,.20),'kappa':(.32,.39)}}
WIDTH_CANDIDATES={'velocity4':(6,4,2),'velocity8':(6,8,2)}
RIDGE_COEF=1e-6

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def tensor_rows(x,y):return torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(y,dtype=torch.float32)

def fit_observed_coef(basis,x,y):
    phi=design(basis,torch.as_tensor(x,dtype=torch.float32)).detach().double()
    target=torch.as_tensor(y,dtype=torch.float64).reshape(-1)
    gram=phi.T@phi;scale=torch.trace(gram)/phi.shape[1]
    eye=torch.eye(phi.shape[1],dtype=phi.dtype)
    w=torch.linalg.solve(gram+RIDGE_COEF*scale.clamp_min(1e-20)*eye,phi.T@target)
    return w.float().detach()

def crossfit_effect_tasks(params_list,seeds,split,basis,code_map=None,fold_n=512):
    """Create paired-fold effects only from independent noisy observed transitions."""
    tasks=[];diag={str(c):{'target_energy_sum':0.,'noise_variance_sum':0.,'pair_count':0} for c in range(4)}
    for bi,(params,bg_seed) in enumerate(zip(params_list,seeds)):
        bgid=f'{split}-{bi:03d}'
        before_folds=[]
        for fold in range(2):
            tr=rollout(params,n=fold_n,seed=int(bg_seed)+3100000+fold*100003,dt=.05,sigma=.0005)
            before_folds.append((rows(tr)[0],rows(tr)[1]))
        for physical in range(4):
            after=apply_program(params,(physical,));after_folds=[]
            for fold in range(2):
                tr=rollout(after,n=fold_n,seed=int(bg_seed)+3200000+physical*10007+fold*100003,dt=.05,sigma=.0005)
                after_folds.append((rows(tr)[0],rows(tr)[1]))
            w_before=[fit_observed_coef(basis,*before_folds[f]) for f in range(2)]
            w_after=[fit_observed_coef(basis,*after_folds[f]) for f in range(2)]
            delta=[w_after[f]-w_before[f] for f in range(2)]
            probes=[np.concatenate((before_folds[f][0],after_folds[f][0]),axis=0) for f in range(2)]
            opaque=int(code_map[physical]) if code_map is not None else physical
            # Cross-fit in both directions: fit fold 0 predicts probes from fold 1,
            # then fit fold 1 predicts fold 0 probes. No paired simulator targets.
            predictions=[]
            for fit_fold,probe_fold in ((0,1),(1,0)):
                probe=torch.as_tensor(probes[probe_fold],dtype=torch.float32)
                tasks.append(ObservedEffectTask(opaque,w_before[fit_fold],delta[fit_fold],probe,
                    bgid,fit_fold,0.0))
                with torch.no_grad():
                    phi=design(basis,probe).detach()
                    predictions.append((phi@delta[fit_fold]).reshape(-1))
            mismatch=float(torch.mean((predictions[0]-predictions[1]).square()))
            energy=float(.5*(predictions[0].square().mean()+predictions[1].square().mean()))
            d=diag[str(physical)];d['target_energy_sum']+=energy;d['noise_variance_sum']+=mismatch/2
            d['pair_count']+=1
    scales={};diagnostics={}
    for physical,d in diag.items():
        energy=d['target_energy_sum']/max(1,d['pair_count']);noise=d['noise_variance_sum']/max(1,d['pair_count'])
        scale=max(energy,noise,1e-12)
        opaque=int(code_map[int(physical)]) if code_map is not None else int(physical)
        scales[opaque]=scale
        diagnostics[physical]={'crossfit_target_energy':energy,'estimated_crossfit_noise_variance':noise,
            'target_to_noise_ratio':energy/max(noise,1e-20),'loss_normalizer_train_only':scale,
            'backgrounds':d['pair_count']}
    return tasks,scales,diagnostics

def reliability(tasks):
    out={}
    by={c:{} for c in range(4)}
    for t in tasks:
        by[t.opaque_code].setdefault(t.background_id,[]).append(t)
    for code,bgs in by.items():
        error=energy=0.
        for pair in bgs.values():
            if len(pair)!=2:continue
            va=[]
            for task in pair:
                va.append(task.delta_coeff)
            error+=float((va[0]-va[1]).square().mean())
            energy+=float(.5*(va[0].square().mean()+va[1].square().mean()))
        out[str(code)]={'crossfit_coefficient_repeatability':float(1.-error/max(energy,1e-30)) if bgs else None,
                        'backgrounds':len(bgs)}
    return out

def write_rows(path,records):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='results/q1/q1_v2_root_repair_dev3')
    ap.add_argument('--seed',type=int,default=20261031);args=ap.parse_args()
    out=ROOT/args.output_dir
    resume=(out/'protocol.json').exists()
    out.mkdir(parents=True,exist_ok=True);(out/'diagnostics').mkdir(exist_ok=True)
    started=time.time();torch.set_num_threads(1)
    old_run=ROOT/'results/q1/q1_v2_operator_retrain_dev1';old_cfg=yaml.safe_load((old_run/'config.resolved.yaml').read_text(encoding='utf-8'))
    if resume:
        root_protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        args.seed=int(root_protocol['seed'])
        train_seeds=root_protocol['new_train']['seeds'];inner_seeds=root_protocol['new_inner_train']['seeds']
        id_seeds=root_protocol['new_development_idlike']['seeds'];ood_seeds=root_protocol['new_development_background_ood']['seeds']
        train_params=[sample_params(np.random.default_rng(s),'train',TRAIN_COUNTS) for s in train_seeds]
        inner_params=[sample_params(np.random.default_rng(s),'train',TRAIN_COUNTS) for s in inner_seeds]
        id_params=[sample_params(np.random.default_rng(s),'train',TRAIN_COUNTS) for s in id_seeds]
        ood_params=[sample_params(np.random.default_rng(s),'dev_ood',OOD_RANGES) for s in ood_seeds]
    else:
        forbidden,sources=gather_forbidden_seeds()
        train_seeds,train_params=collect_backgrounds(40,args.seed+1,forbidden,'train',TRAIN_COUNTS);forbidden.update(train_seeds)
        inner_seeds,inner_params=collect_backgrounds(8,args.seed+2,forbidden,'train',TRAIN_COUNTS);forbidden.update(inner_seeds)
        id_seeds,id_params=collect_backgrounds(16,args.seed+3,forbidden,'train',TRAIN_COUNTS);forbidden.update(id_seeds)
        ood_seeds,ood_params=collect_backgrounds(16,args.seed+4,forbidden,'dev_ood',OOD_RANGES)
    projection_src=out/'diagnostics'/'projection_run04'
    if not (projection_src/'direct_effect_projection.json').exists():
        raise FileNotFoundError('successful Phase 1 direct effect projection run is required')
    # Branch rule from the attachment: U4 is intermediate; compare width 4 and 8
    # using inner-train-only diagnostics, then freeze one before fresh evaluation.
    projection=json.loads((projection_src/'direct_effect_projection.json').read_text(encoding='utf-8'))
    branch_metrics={}
    for regime in ('IDLIKE','BACKGROUND_OOD'):
        branch_metrics[regime]={}
        for budget in (512,1024):
            key=f'{regime}|U4|budget={budget}'
            branch_metrics[regime][str(budget)]=projection['metrics'][key]['B_paired_projection']['mean']
    new_root_protocol={'run':'Q1 V2 comprehensive root repair','branch_sha_before_training':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_run':'results/q1/q1_v2_operator_retrain_dev1','diagnostic_projection':'diagnostics/projection_run04',
      'seed':args.seed,'train_backgrounds':40,'inner_train_backgrounds':8,'development_backgrounds_per_regime':16,
      'dt':.05,'sigma':.0005,'old_support':96,'adaptation_trajectory':100,'query_trajectory':256,
      'N':[5,10,20,50,100],'candidate_program_count':21,'operator_seeds':old_cfg['optimizer_seeds'],
      'fit_transition_noise_preserved':True,'new_train':{'seeds':train_seeds},'new_inner_train':{'seeds':inner_seeds},
      'new_development_idlike':{'seeds':id_seeds},'new_development_background_ood':{'seeds':ood_seeds},
      'direct_projection_U4_branch_metrics':branch_metrics,'basis_capacity_branch':'intermediate 0.5-0.8; select smallest width by train-inner evidence',
      'original_test_splits_used':False,'original_Q1_thresholds_modified':False,
      'collision_audit_sources':sources if not resume else root_protocol.get('collision_audit_sources',[]),'previous_baseline_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in [old_run/'basis_pretrained.pt',old_run/'operator_seed4201.pt',old_run/'operator_seed4202.pt',old_run/'metrics.json',ROOT/'results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt']}}
    if not resume:
      root_protocol=new_root_protocol
      (out/'protocol.json').write_text(json.dumps(root_protocol,indent=2),encoding='utf-8')
    else:
      root_protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if not (out/'config.yaml').exists():
      (out/'config.yaml').write_text(yaml.safe_dump({'seed':args.seed,'dt':.05,'sigma':.0005,'old_support':96,
      'adaptation_trajectory':100,'query_trajectory':256,'budgets':[5,10,20,50,100],
      'train_backgrounds':40,'inner_backgrounds':8,'development_backgrounds_per_regime':16,
      'basis_width_candidates':{'velocity4':[6,4,2],'velocity8':[6,8,2]},'hidden':32,'basis_epochs':18,
      'operator_epochs':4,'learning_rate':.001,'operator_regularization':.0001,'effect_weight':1.0,
      'effect_fit_fold_transitions':512,'composition_training_pairs':old_cfg['composition_training_pairs'],
      'operator_seeds':old_cfg['optimizer_seeds'],'length_penalty_grid':[0.0,.01,.1]},sort_keys=False),encoding='utf-8')
    # Snapshot old splits, output manifests and source/checkpoint hashes before generation/fitting.
    pre_sources=[ROOT/'q1_v2/basis.py',ROOT/'q1_v2/bayes.py',ROOT/'q1_v2/operators.py',ROOT/'q1_v2/inference.py',ROOT/'q1_v2/training.py',ROOT/'scripts/q1_v2_pipeline.py',ROOT/'scripts/q1_v2_root_repair.py',ROOT/'q1/simulator.py']
    if not (out/'source_manifest.json').exists():
      (out/'source_manifest.json').write_text(json.dumps({'branch':root_protocol['branch_sha_before_training'],
      'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in pre_sources},
      'historical_run_artifacts_sha256':root_protocol['previous_baseline_hashes'],
      'previous_inspected_split_manifests':{k:sha(ROOT/k) for k in [
        'results/q1/q1_v2_operator_retrain_dev1/split_manifest.json',
        'results/q1/q1_v2_bayesian_operators_dev1/split_manifest.json'] if (ROOT/k).exists()}},indent=2),encoding='utf-8')

    # Build weak-label episodes once; architecture candidates see identical train/inner observations.
    fit_eps,inner_eps,op_eps,inner_op=make_training_data(train_params,inner_params,old_cfg)
    basis_candidates={};candidate_metrics={}
    for name,widths in WIDTH_CANDIDATES.items():
        basis=PartitionedBasis(widths=widths,hidden=32)
        stats=fit_basis(basis,fit_eps,sigma2=.0005**2,seed=args.seed+(410 if name=='velocity4' else 810),
            epochs=18,batch_size=8,lr=.001,inner_episodes=inner_eps)
        validation=evaluate_basis_nll(basis,inner_eps,stats['prior_mean'],stats['prior_covariance'].diag().log(),.0005**2)
        train_tasks,train_scales,train_crossfit=crossfit_effect_tasks(train_params,train_seeds,'train',basis)
        inner_tasks,_,inner_crossfit=crossfit_effect_tasks(inner_params,inner_seeds,'inner',basis)
        rep_train=reliability(train_tasks);rep_inner=reliability(inner_tasks)
        candidate_metrics[name]={'widths':widths,'inner_predictive_nll':validation,'selected_basis_epoch':stats['selected_epoch'],
          'train_effect_repeatability':rep_train,'inner_effect_repeatability':rep_inner,
          'inner_u4_repeatability_mean':float(np.mean([rep_inner[str(c)]['crossfit_coefficient_repeatability'] for c in (3,)])),
          'curve':stats['curve'],'train_effect_normalization':train_crossfit,'inner_effect_noise_diagnostic':inner_crossfit}
        checkpoint={'basis':{k:v.detach().cpu() for k,v in basis.state_dict().items()},'widths':widths,
          'prior_mean':stats['prior_mean'].detach().cpu(),'prior_covariance':stats['prior_covariance'].detach().cpu(),
          'sigma2':.0005**2,'selected_epoch':stats['selected_epoch']}
        candidate_path=out/f'basis_candidate_{name}.pt'
        if candidate_path.exists():
            suffix=2
            while (out/f'basis_candidate_{name}_attempt{suffix:02d}.pt').exists():suffix+=1
            candidate_path=out/f'basis_candidate_{name}_attempt{suffix:02d}.pt'
        torch.save(checkpoint,candidate_path)
        basis_candidates[name]=(basis,stats,train_tasks,train_scales,train_crossfit,inner_tasks,inner_crossfit)
        print(f"Basis candidate {name} inner NLL={validation:.6g}",flush=True)
    nll4=candidate_metrics['velocity4']['inner_predictive_nll'];nll8=candidate_metrics['velocity8']['inner_predictive_nll']
    improvement=(nll4-nll8)/max(abs(nll4),1e-9)
    u12_stable=all(candidate_metrics['velocity8']['inner_effect_repeatability'][str(c)]['crossfit_coefficient_repeatability'] >=
        candidate_metrics['velocity4']['inner_effect_repeatability'][str(c)]['crossfit_coefficient_repeatability']-.1 for c in (0,1))
    chosen='velocity8' if improvement>=.02 and u12_stable else 'velocity4'
    (out/'basis_capacity_selection.json').write_text(json.dumps({'train_inner_only':True,
      'predeclared_selector':'choose velocity width 8 only if inner predictive NLL improves by at least 2% and U1/U2 cross-fit coefficient repeatability does not drop by more than 0.10; otherwise choose width 4',
      'inner_nll_relative_improvement_width8':improvement,'u1_u2_repeatability_guard':u12_stable,
      'selected_candidate':chosen,'candidate_metrics':candidate_metrics},indent=2),encoding='utf-8')
    basis,stats,train_tasks,train_scales,train_crossfit,inner_tasks,inner_crossfit=basis_candidates[chosen]
    widths=WIDTH_CANDIDATES[chosen];k=sum(widths);sigma2=.0005**2
    torch.save({'basis':{k0:v.detach().cpu() for k0,v in basis.state_dict().items()},'widths':widths,
      'prior_mean':stats['prior_mean'].detach().cpu(),'prior_covariance':stats['prior_covariance'].detach().cpu(),
      'sigma2':sigma2,'selected_epoch':stats['selected_epoch']},out/'basis_pretrained.pt')
    # Freeze randomized opaque labels for all training/inner learner interfaces.
    physical_order=np.random.default_rng(args.seed+900).permutation(4).tolist()
    code_map={physical:int(physical_order[physical]) for physical in range(4)}
    train_ops=[ObservedEpisode(e.support_x,e.support_y,e.query_x,e.query_y,tuple(code_map[c] for c in e.opaque_program)) for e in op_eps]
    inner_ops=[ObservedEpisode(e.support_x,e.support_y,e.query_x,e.query_y,tuple(code_map[c] for c in e.opaque_program)) for e in inner_op]
    def remap_tasks(ts):
        return [ObservedEffectTask(code_map[int(t.opaque_code)],t.before_coeff,t.delta_coeff,t.probe_x,t.background_id,t.fold,t.noise_variance) for t in ts]
    train_tasks=remap_tasks(train_tasks);inner_tasks=remap_tasks(inner_tasks)
    train_scales={code_map[int(c)]:v for c,v in train_scales.items()}
    inner_scales=train_scales.copy()
    selection_codes={f'U{i+1}':code_map[i] for i in range(4)}
    nll_banks={};effect_banks={};operator_training={}
    for seed in old_cfg['optimizer_seeds']:
        baseline,bd=fit_operator_bank(basis,stats['prior_mean'],stats['prior_covariance'],train_ops,
          sigma2=sigma2,seed=seed,epochs=4,lr=.001,regularization=1e-4,inner_episodes=inner_ops,
          normalize_nll_per_observation=True)
        effect,ed=fit_operator_bank_effect_aware(basis,stats['prior_mean'],stats['prior_covariance'],train_ops,
          train_tasks,effect_scales=train_scales,sigma2=sigma2,seed=seed,epochs=4,lr=.001,
          regularization=1e-4,effect_weight=1.0,inner_episodes=inner_ops,inner_effect_tasks=inner_tasks,
          inner_effect_scales=inner_scales,selection_code_groups=selection_codes)
        nll_banks[seed]=baseline;effect_banks[seed]=effect
        torch.save({'state_dict':baseline.state_dict(),'widths':widths,'opaque_code_map':code_map,
          'method':'normalized NLL only'},out/f'operator_seed{seed}_nll_only.pt')
        torch.save({'state_dict':effect.state_dict(),'widths':widths,'opaque_code_map':code_map,
          'method':'normalized NLL plus cross-fitted observed effect'},out/f'operator_seed{seed}_effect_aware.pt')
        operator_training[str(seed)]={'nll_only':bd,'effect_aware':ed}
        print(f'Operator variants trained for seed {seed}',flush=True)
    (out/'operator_training.json').write_text(json.dumps({'opaque_code_map':{str(k):v for k,v in code_map.items()},
      'train_only_effect_normalizers':{str(k):v for k,v in train_scales.items()},
      'crossfit_train_noise_diagnostics':train_crossfit,'crossfit_inner_noise_diagnostics':inner_crossfit,
      'per_seed':operator_training},indent=2),encoding='utf-8')
    torch.save({'basis':{k:v.detach().cpu() for k,v in basis.state_dict().items()},'widths':widths,
      'prior_mean':stats['prior_mean'].detach().cpu(),'prior_covariance':stats['prior_covariance'].detach().cpu(),
      'sigma2':sigma2,'selected_epoch':stats['selected_epoch']},out/'basis_pretrained.pt')
    # Inner-only length penalty selection on previously permitted programs.
    penalty_grid=[0.,.01,.1];allowed=[()] + [(i,) for i in range(4)] + [tuple(p) for p in old_cfg['composition_training_pairs']]
    penalty_scores=[]
    inverse={v:k for k,v in code_map.items()}
    for penalty in penalty_grid:
        ok=[]
        for bi,bg in enumerate(inner_params):
          for pi,program in enumerate(allowed):
            old,adapt,_=stage_pair(bg,program,seed=870000+bi*1009+pi,n_support=96,n_query=100,dt=.05,sigma=.0005)
            classes=functional_classes(bg,dt=.05)[0]
            ox,oy=rows(old);ax,ay=rows(adapt)
            for seed,bank in effect_banks.items():
              mean=stats['prior_mean'];cov=stats['prior_covariance']
              post=gaussian_posterior(design(basis,torch.as_tensor(ox)),torch.as_tensor(oy),mean,cov,sigma2)
              candidates=[tuple(code_map[c] for c in p) for p in allowed]
              ranked=rank_programs(bank,post,candidates,design(basis,torch.as_tensor(ax[:20])),torch.as_tensor(ay[:20]),sigma2,penalty)
              pred=tuple(inverse[c] for c in ranked[0][0]);ok.append(classes[PROGRAMS.index(pred)]==classes[PROGRAMS.index(program)])
        penalty_scores.append({'penalty':penalty,'inner_class_accuracy':float(np.mean(ok))})
    length_penalty=max(penalty_scores,key=lambda r:(r['inner_class_accuracy'],-r['penalty']))['penalty']
    (out/'length_penalty_selection.json').write_text(json.dumps({'inner_train_only':True,'selected':length_penalty,'candidates':penalty_scores},indent=2),encoding='utf-8')
    # Fresh, frozen, low-excitation development evaluation; no query labels enter rankings.
    eval_rows=[];bg_labels={};programs=list(PROGRAMS)
    for regime,bgs,seeds in [('IDLIKE_DEVELOPMENT',id_params,id_seeds),('BACKGROUND_OOD_DEVELOPMENT',ood_params,ood_seeds)]:
      for bi,bg in enumerate(bgs):
        bgid=f'{regime}-{bi:03d}';bg_seed=int(seeds[bi]);classes=functional_classes(bg,dt=.05)[0]
        bg_labels[bgid]=int(bi)
        print(f'Evaluating {regime} {bi+1}/{len(bgs)}',flush=True)
        for pi,true_program in enumerate(programs):
          old,adapt,query=stage_pair(bg,true_program,seed=bg_seed+300000+pi*7919,
            n_support=96,n_query=256,dt=.05,sigma=.0005)
          ox,oy=rows(old);ax,ay=rows(adapt);qx,qy=rows(query)
          oldpost={};qphi=design(basis,torch.as_tensor(qx));target=torch.as_tensor(qy).reshape(-1)
          for seed in old_cfg['optimizer_seeds']:
            oldpost[seed]=gaussian_posterior(design(basis,torch.as_tensor(ox)),torch.as_tensor(oy),
                stats['prior_mean'],stats['prior_covariance'],sigma2)
          for variant,banks in [('NLL_ONLY',nll_banks),('EFFECT_AWARE',effect_banks)]:
           for seed,bank in banks.items():
            for n in old_cfg['budgets']:
              cand=[tuple(code_map[c] for c in p) for p in programs]
              phi=design(basis,torch.as_tensor(ax[:n]));y=torch.as_tensor(ay[:n])
              ranked=rank_programs(bank,oldpost[seed],cand,phi,y,sigma2,length_penalty)
              point=rank_programs(bank,oldpost[seed],cand,phi,y,sigma2,length_penalty,point_estimate=True)
              inv={v:k for k,v in code_map.items()}
              pred=tuple(inv[c] for c in ranked[0][0]);point_pred=tuple(inv[c] for c in point[0][0])
              am,bb=composed(bank,ranked[0][0]);pred_y=qphi@(am@oldpost[seed].mean+bb)
              mse=float(torch.mean((pred_y.reshape(-1)-target)**2))
              effect_skill=None;true_energy=None
              if len(true_program)==1:
                code=true_program[0];am1,bb1=composed(bank,(code_map[code],))
                pred_effect=(qphi@((am1@oldpost[seed].mean+bb1)-oldpost[seed].mean)).reshape(-1,2).detach().numpy()
                true_effect=response(apply_program(bg,true_program),qx,dt=.05)-response(bg,qx,dt=.05)
                true_energy=float(np.mean(true_effect**2))
                effect_skill=1-float(np.mean((pred_effect-true_effect)**2))/max(true_energy,1e-30)
              eval_rows.append({'regime':regime,'background_id':bgid,'background_seed':bg_seed,
                'optimizer_seed':seed,'model':variant,'true_program':json.dumps(true_program),'N':n,
                'predicted_program':json.dumps(pred),'point_predicted_program':json.dumps(point_pred),
                'functional_class_correct':int(classes[PROGRAMS.index(pred)]==classes[pi]),
                'point_functional_class_correct':int(classes[PROGRAMS.index(point_pred)]==classes[pi]),
                'query_delta_v_mse':mse,'effect_skill':effect_skill,'true_effect_mean_square':true_energy})
    write_rows(out/'per_background.csv',eval_rows)
    # Save raw per-program aggregates now; final report adds paired baseline/bootstrap tables.
    prog_rows=[]
    for key in sorted({(r['regime'],r['model'],r['N'],r['true_program']) for r in eval_rows}):
      cell=[r for r in eval_rows if (r['regime'],r['model'],r['N'],r['true_program'])==key]
      prog_rows.append({'regime':key[0],'model':key[1],'N':key[2],'true_program':key[3],
        'n_background_seed_rows':len(cell),'functional_class_accuracy':float(np.mean([r['functional_class_correct'] for r in cell])),
        'point_functional_class_accuracy':float(np.mean([r['point_functional_class_correct'] for r in cell])),
        'mean_query_delta_v_mse':float(np.mean([r['query_delta_v_mse'] for r in cell]))})
    write_rows(out/'per_program.csv',prog_rows)
    # Legacy evaluator adds old V2 and M4 on the exact same generated cases/seeds.
    (out/'config.resolved.yaml').write_text(yaml.safe_dump({'run_id':'q1_v2_root_repair_dev3','seed':args.seed,'dt':.05,'sigma':.0005,
      'K':k,'blocks':list(widths),'basis_widths':list(widths),'hidden':32,'old_support':96,'adaptation_trajectory':100,
      'query_trajectory':256,'budgets':[5,10,20,50,100],'optimizer_seeds':old_cfg['optimizer_seeds'],
      'composition_training_pairs':old_cfg['composition_training_pairs']},sort_keys=False),encoding='utf-8')
    provenance={'runtime_seconds_before_legacy':time.time()-started,'device':'cpu','test_split_used':False,
      'source_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in pre_sources},
      'checkpoint_hashes':{p.name:sha(p) for p in out.glob('*.pt')},'optimizer_seeds':old_cfg['optimizer_seeds'],
      'evaluation_rows':len(eval_rows),'basis_widths':widths,'basis_candidate':chosen}
    (out/'provenance.prelegacy.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(f'Q1 V2 root repair training/evaluation complete before legacy; {len(eval_rows)} rows in {time.time()-started:.1f}s',flush=True)

if __name__=='__main__':main()
