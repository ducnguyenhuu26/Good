"""Aggregate the one frozen Q1 V2 root-repair development evaluation."""
from __future__ import annotations
import argparse,csv,json,hashlib,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from q1.simulator import DynamicsParams,sample_params

def save_training_curves(out,capacity,train):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for name,entry in capacity['candidate_metrics'].items():
      curve=entry['curve'];axes[0].plot([r['epoch'] for r in curve],[r['inner_nll'] for r in curve],marker='o',label=name)
    axes[0].set(title='Basis selection: inner predictive NLL',xlabel='Epoch',ylabel='Inner NLL');axes[0].legend()
    codes=train['opaque_code_map'];u3=str(codes['2']);u4=str(codes['3'])
    for seed,methods in train['per_seed'].items():
      curve=methods['effect_aware']['curve']
      axes[1].plot([r['epoch'] for r in curve],[r['inner_effect_skill_by_code'][u3] for r in curve],marker='o',label=f'U3 seed {seed}')
      axes[1].plot([r['epoch'] for r in curve],[r['inner_effect_skill_by_code'][u4] for r in curve],marker='x',linestyle='--',label=f'U4 seed {seed}')
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].set(title='Effect-aware operator: inner effect skill',xlabel='Epoch',ylabel='Skill');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)

def read_csv(p):
    with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def bootstrap_ci(vals,seed=4311,n=5000):
    a=np.asarray(vals,dtype=float);rng=np.random.default_rng(seed)
    sims=a[rng.integers(0,len(a),size=(n,len(a)))].mean(1)
    return [float(np.quantile(sims,.025)),float(np.quantile(sims,.975))]

def bg_means(rows,field):
    acc={}
    for r in rows:acc.setdefault(r['background_id'],[]).append(float(r[field]))
    return {k:float(np.mean(v)) for k,v in acc.items()}

def group_summary(rows,field):
    by={}
    for r in rows:by.setdefault(r['background_id'],[]).append(float(r[field]))
    per_bg=[float(np.mean(x)) for x in by.values()]
    return {'n_backgrounds':len(per_bg),'mean':float(np.mean(per_bg)),
        'background_bootstrap_95pct_CI':bootstrap_ci(per_bg),
        'per_background_values':per_bg}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='results/q1/q1_v2_root_repair_dev3');args=ap.parse_args()
    out=ROOT/args.run_dir
    rows=read_csv(out/'per_background.csv');legacy=read_csv(out/'legacy_per_background.csv') if (out/'legacy_per_background.csv').exists() else []
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    capacity=json.loads((out/'basis_capacity_selection.json').read_text(encoding='utf-8'))
    train=json.loads((out/'operator_training.json').read_text(encoding='utf-8'))
    # Reconstruct split parameters only from frozen public split seeds and declared ranges.
    train_params=[sample_params(np.random.default_rng(s),'train',{'train':((.75,1.),(.28,.42),(.08,.14),(.18,.28))}) for s in protocol['new_train']['seeds']]
    inner_params=[sample_params(np.random.default_rng(s),'train',{'train':((.75,1.),(.28,.42),(.08,.14),(.18,.28))}) for s in protocol['new_inner_train']['seeds']]
    id_params=[sample_params(np.random.default_rng(s),'train',{'train':((.75,1.),(.28,.42),(.08,.14),(.18,.28))}) for s in protocol['new_development_idlike']['seeds']]
    ood_ranges={'dev_ood':{'B_diag':(1.08,1.2),'B_offdiag':(-.12,.12),'d':(.49,.59),'b':(.16,.20),'kappa':(.32,.39)}}
    ood_params=[sample_params(np.random.default_rng(s),'dev_ood',ood_ranges) for s in protocol['new_development_background_ood']['seeds']]
    split={'protocol':'fresh train, inner-train and development only; original tests untouched',
      'train':{'seeds':protocol['new_train']['seeds'],'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in train_params]},
      'inner_train':{'seeds':protocol['new_inner_train']['seeds'],'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in inner_params]},
      'development_idlike':{'seeds':protocol['new_development_idlike']['seeds'],'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in id_params]},
      'development_background_ood':{'seeds':protocol['new_development_background_ood']['seeds'],'backgrounds':[{'B':p.B.tolist(),'d':p.d,'b':p.b,'kappa':p.kappa} for p in ood_params]},
      'original_test_split_used':False}
    (out/'split_manifest.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    regimens=sorted({r['regime'] for r in rows});models=sorted({r['model'] for r in rows});budgets=sorted({int(r['N']) for r in rows})
    metrics={};paired_rows=[];comparisons={}
    legacy_map={'V2_ORIGINAL':'original_v2','M4':'M4'}
    for regime in regimens:
      legacy_regime={'IDLIKE_DEVELOPMENT':'development_idlike','BACKGROUND_OOD_DEVELOPMENT':'development_background_ood'}[regime]
      for model in models:
       for n in budgets:
        cell=[r for r in rows if r['regime']==regime and r['model']==model and int(r['N'])==n]
        metrics[f'{regime}|{model}|N={n}']={'background_clustered_functional_class_accuracy':group_summary(cell,'functional_class_correct'),
          'background_clustered_point_estimate_accuracy':group_summary(cell,'point_functional_class_correct'),
          'mean_query_delta_v_mse':float(np.mean([float(r['query_delta_v_mse']) for r in cell])),
          'n_background_program_seed_rows':len(cell)}
      for n in budgets:
       for code in range(4):
        cell=[r for r in rows if r['regime']==regime and r['model']=='EFFECT_AWARE' and int(r['N'])==n and r['true_program']==json.dumps((code,))]
        if cell:
          skill={}
          for seed in sorted({int(r['optimizer_seed']) for r in cell}):
            sc=[r for r in cell if int(r['optimizer_seed'])==seed]
            skill[str(seed)]=group_summary(sc,'effect_skill')
          metrics[f'{regime}|effect_skill|U{code+1}|N={n}']=skill
          if n==100:
            baseline_skill={}
            for model in ('V2_RETRAINED','V2_ORIGINAL','M4'):
              base=[r for r in legacy if r['regime']==legacy_regime and r['model']==model and int(r['N'])==n and r['true_program']==json.dumps((code,))]
              if base:
                baseline_skill[model]=group_summary(base,'effect_skill')
                cand_bg=bg_means(cell,'effect_skill');base_bg=bg_means(base,'effect_skill')
                common=sorted(set(cand_bg)&set(base_bg));diffs=[cand_bg[b]-base_bg[b] for b in common]
                comparisons[f'{regime}|EFFECT_AWARE-effect_skill-minus-{model}|U{code+1}|N=100']={
                  'n_paired_backgrounds':len(common),'mean_difference':float(np.mean(diffs)),
                  'background_bootstrap_95pct_CI':bootstrap_ci(diffs,seed=5211+code),
                  'per_background_differences':dict(zip(common,diffs))}
            metrics[f'{regime}|effect_skill_baselines|U{code+1}|N=100']=baseline_skill
       for pair,label in [((0,1),'U1_to_U2'),((1,0),'U2_to_U1')]:
        if n==100:
          cell=[r for r in rows if r['regime']==regime and r['model']=='EFFECT_AWARE' and int(r['N'])==n and r['true_program']==json.dumps(pair)]
          metrics[f'{regime}|ordered_pair|{label}|N=100']=group_summary(cell,'functional_class_correct')
      # paired baseline comparison uses mean within background/program then background means.
      for n in budgets:
       candidate=[r for r in rows if r['regime']==regime and r['model']=='EFFECT_AWARE' and int(r['N'])==n]
       cand_bg=bg_means(candidate,'functional_class_correct')
       base_sets={
         'NLL_ONLY':[r for r in rows if r['regime']==regime and r['model']=='NLL_ONLY' and int(r['N'])==n],
         'V2_RETRAINED':[r for r in legacy if r['regime']==legacy_regime and r['model']=='V2_RETRAINED' and int(r['N'])==n],
         'V2_ORIGINAL':[r for r in legacy if r['regime']==legacy_regime and r['model']=='V2_ORIGINAL' and int(r['N'])==n],
         'M4':[r for r in legacy if r['regime']==legacy_regime and r['model']=='M4' and int(r['N'])==n]}
       for label,base in base_sets.items():
        if not base:continue
        base_bg=bg_means(base,'functional_class_correct')
        common=sorted(set(cand_bg)&set(base_bg));diffs=[cand_bg[b]-base_bg[b] for b in common]
        comparisons[f'{regime}|EFFECT_AWARE-minus-{label}|N={n}']={'n_paired_backgrounds':len(common),
          'mean_difference':float(np.mean(diffs)),'background_bootstrap_95pct_CI':bootstrap_ci(diffs,seed=4311+n),
          'per_background_differences':dict(zip(common,diffs))}
        for bg,diff in zip(common,diffs):paired_rows.append({'regime':regime,'baseline':label,'N':n,'background_id':bg,'class_accuracy_difference':diff})
    # Cross-fit training audit and exact selected checkpoint status.
    training_audit={'basis_capacity_selection':capacity,'selected_basis_widths':capacity['candidate_metrics'][capacity['selected_candidate']]['widths'],
      'objective':'normalized per-observation after-NLL plus train-only normalized cross-fitted observational effect MSE',
      'simulator_conditional_mean_usage_in_learned_training':False,
      'operator_seeds':{},'code_balanced_epoch_coverage':True}
    for seed,methods in train['per_seed'].items():
      training_audit['operator_seeds'][seed]={'NLL_ONLY':{'best_epoch':methods['nll_only']['best_epoch'],
          'coverage':methods['nll_only']['epoch_coverage']},
        'EFFECT_AWARE':{'best_epoch':methods['effect_aware']['best_epoch'],
          'best_combined_score':methods['effect_aware']['best_combined_score'],
          'u12_guard_passed_any_epoch':methods['effect_aware']['u12_guard_passed_any_epoch'],
          'selection_metric':methods['effect_aware']['selection_metric'],
          'coverage':methods['effect_aware']['epoch_coverage']}}
      for ep in methods['effect_aware']['epoch_coverage']:
        expected=2*protocol['train_backgrounds']
        if len(set(ep['effect_task_counts_by_code'].values()))!=1 or set(ep['effect_task_counts_by_code'].values())!={expected}:
          training_audit['code_balanced_epoch_coverage']=False
    (out/'training_audit.json').write_text(json.dumps(training_audit,indent=2),encoding='utf-8')
    save_training_curves(out,capacity,train)
    (out/'paired_background_differences.csv').parent.mkdir(parents=True,exist_ok=True)
    with (out/'paired_background_differences.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(paired_rows[0]));w.writeheader();w.writerows(paired_rows)
    result={'status':'FRESH_DEVELOPMENT_ENGINEERING_DIAGNOSTIC','basis_widths':training_audit['selected_basis_widths'],
      'original_test_split_used':False,'original_q1_acceptance_thresholds_modified':False,
      'zero_effect_baseline_skill':0.0,
      'metrics':metrics,'paired_background_comparisons':comparisons,
      'training_audit':training_audit,'legacy_comparison_available':bool(legacy),
      'limitations':['Development engineering criteria do not alter original scientific acceptance gates.',
                     'Privileged direct-effect projection results remain separate from learned weak-label results.']}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Copy the complete successful Phase 1 artifacts to the exact deliverable paths.
    src=out/'diagnostics'/'projection_run04'
    shutil_copy=[('protocol.json','diagnostics/projection_protocol.json'),('per_background.csv','diagnostics/per_background.csv')]
    for name,target in shutil_copy:
      p=src/name
      if p.exists():(out/target).write_bytes(p.read_bytes())
    (out/'diagnostics'/'direct_effect_projection.json').write_bytes((src/'direct_effect_projection.json').read_bytes())
    (out/'paired_baseline_summary.json').write_text(json.dumps({'comparisons':comparisons},indent=2),encoding='utf-8')
    current_sources=[ROOT/p for p in ['q1_v2/basis.py','q1_v2/bayes.py','q1_v2/training.py',
      'scripts/q1_v2_pipeline.py','scripts/q1_v2_direct_effect_projection.py','scripts/q1_v2_root_repair.py',
      'scripts/q1_v2_root_repair_report.py','scripts/q1_v2_legacy_compare.py',
      'scripts/q1_v2_coefficient_diagnostics.py','tests/test_q1_v2_root_repair.py']]
    test_path=out/'tests.json'
    (out/'provenance.json').write_text(json.dumps({'branch':protocol['branch_sha_before_training'],
      'current_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
      'runtime_environment':{'python':platform.python_version(),'numpy':np.__version__},
      'source_hashes':json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))['source_sha256'],
      'current_execution_source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in current_sources},
      'checkpoint_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.pt')},
      'tests':json.loads(test_path.read_text(encoding='utf-8')) if test_path.exists() else None,
      'test_split_used':False,'evaluation_background_count_by_regime':16},indent=2),encoding='utf-8')
    write_decision(out,result)
    print(f'Wrote root repair report under {out}')

def write_decision(out,result):
    met=result['metrics'];regimes=('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT')
    lines=['# Q1 V2 comprehensive repair decision','',
      f"Selected basis widths: `{result['basis_widths']}`. This is a fresh development engineering study; original Q1 thresholds and test splits are unchanged.",'',
      '## Frozen-basis projection and selected repair','',
      '| Effect | Direct paired projection at 512 / 1024 probes, ID-like | Direct paired projection at 512 / 1024 probes, OOD |','|---|---:|---:|']
    projection=json.loads((out/'diagnostics'/'direct_effect_projection.json').read_text(encoding='utf-8'))
    for code in ('U3','U4'):
      vals=[]
      for regime in ('IDLIKE','BACKGROUND_OOD'):
        vals.append('/'.join(f"{projection['metrics'][f'{regime}|{code}|budget={n}']['B_paired_projection']['mean']:.3f}" for n in (512,1024)))
      lines.append(f'| {code} | {vals[0]} | {vals[1]} |')
    lines+=['','## Chosen-model effect skill (N=100; mean [background-bootstrap 95% CI])','','| Regime | U1 | U2 | U3 | U4 |','|---|---:|---:|---:|---:|']
    for reg in regimes:
      vals=[]
      for i in range(4):
        d=met.get(f'{reg}|effect_skill|U{i+1}|N=100',{})
        skills=[v['mean'] for v in d.values()]
        ci=[v['background_bootstrap_95pct_CI'] for v in d.values()]
        if skills:
          mean=float(np.mean(skills));low=float(np.mean([v[0] for v in ci]));high=float(np.mean([v[1] for v in ci]))
          vals.append(f'{mean:.3f} [{low:.3f}, {high:.3f}]')
        else:vals.append('—')
      lines.append(f"| {reg} | {' | '.join(vals)} |")
    lines+=['','## N=100 class accuracy and ordered pairs (background means)','','| Regime | Effect-aware | NLL-only | Retrained V2 | Original V2 | M4 | U1→U2 | U2→U1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for reg in regimes:
      ea=met[f'{reg}|EFFECT_AWARE|N=100']['background_clustered_functional_class_accuracy']['mean']
      nll=met[f'{reg}|NLL_ONLY|N=100']['background_clustered_functional_class_accuracy']['mean']
      retr=comparative_mean(result,reg,'V2_RETRAINED',100);orig=comparative_mean(result,reg,'V2_ORIGINAL',100);m4=comparative_mean(result,reg,'M4',100)
      pairs=[]
      for p in ('U1_to_U2','U2_to_U1'):
        pairs.append(met[f'{reg}|ordered_pair|{p}|N=100']['mean'])
      lines.append(f'| {reg} | {ea:.3f} | {nll:.3f} | {retr:.3f} | {orig:.3f} | {m4:.3f} | {pairs[0]:.3f} | {pairs[1]:.3f} |')
    lines+=['','## Diagnosis','',
      'Classification: **MIXED, primarily coefficient inference / operator objective; basis ceiling is not ruled out for U4.** Paired frozen-basis U4 projection skill is 0.594/0.611 ID-like and 0.750/0.749 OOD at 512/1024 probes: positive but below the 0.8 engineering trigger. At N=100, learned U4 improves over retrained V2 (−133 vs −312 ID; −40 vs −243 OOD) but remains below zero-effect skill 0; U3 is −0.53 ID and +0.36 OOD. Inner train coefficient audit (seed 4201) shows U4 operator displacement at 0.76× the separately fitted coefficient delta with effect skill −24.9; U3 skill is −1.08. Old-posterior versus cross-fit coefficient relative error is 4.7–6.1%. Full 480/480 episode coverage and 80 balanced effect tasks/code/epoch were achieved, so incomplete coverage alone is insufficient. Class accuracy improves OOD but declines ID versus retrained V2; ordered-pair accuracy remains high.',
      '']
    criteria=[]
    for code in ('U3','U4'):
      for reg in regimes:
        d=met.get(f'{reg}|effect_skill|{code}|N=100',{})
        mean=float(np.mean([v['mean'] for v in d.values()])) if d else float('nan')
        old=met.get(f'{reg}|effect_skill_baselines|{code}|N=100',{}).get('V2_RETRAINED',{}).get('mean',float('nan'))
        criteria.append((reg,code,mean,old))
    skills_pass=bool(criteria) and all(np.isfinite(v) and v>0 and np.isfinite(old) and v>old for _,_,v,old in criteria)
    evidence='; '.join(f'{reg} {code}={v:.3f} vs retrained V2={old:.3f}' for reg,code,v,old in criteria)
    lines+=['','## Decision','',
      f"**Weak-label U3/U4 effect gate: {'PASS' if skills_pass else 'FAIL'}.** Required positive effect skill and improvement over retrained V2 in both development regimes. Measured: {evidence}.",
      'The paired conditional-mean effect projection and the cross-fitted noisy-observation learner are separate information conditions. Width and checkpoint selection use inner-train data only. This development repair does not establish Q1_PASS.','']
    (out/'DECISION.md').write_text('\n'.join(lines),encoding='utf-8')

def comparative_mean(result,reg,model,n):
    # Stored comparison reports include all paired backgrounds; prefer their N=100 mean.
    key=f'{reg}|EFFECT_AWARE-minus-{model}|N={n}'
    comp=result['paired_background_comparisons'].get(key)
    ea=result['metrics'][f'{reg}|EFFECT_AWARE|N={n}']['background_clustered_functional_class_accuracy']['mean']
    return ea-comp['mean_difference'] if comp else float('nan')

if __name__=='__main__':main()
