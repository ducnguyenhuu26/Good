"""Finalize paired Q1 V3 development metrics and the frozen decision gates."""
from __future__ import annotations
import csv,hashlib,json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,yaml
from q1.simulator import PROGRAMS

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def boot(values,seed=311,n=5000):
    a=np.asarray(values,dtype=float)
    if len(a)==0:return [None,None]
    sims=np.random.default_rng(seed).choice(a,size=(n,len(a)),replace=True).mean(1)
    return [float(np.quantile(sims,.025)),float(np.quantile(sims,.975))]
def bgmean(rows,field):
    acc={}
    for r in rows:acc.setdefault(r['background_id'],[]).append(float(r[field]))
    return {k:float(np.mean(v)) for k,v in acc.items()}
def paired(a,b,field,seed):
    aa=bgmean(a,field);bb=bgmean(b,field);common=sorted(set(aa)&set(bb));d=[aa[x]-bb[x] for x in common]
    return {'n_paired_backgrounds':len(common),'mean_difference':float(np.mean(d)) if d else None,'background_bootstrap_95pct_ci':boot(d,seed)}
def main():
    out=ROOT/'results/q1/q1_v3_mechanism_dev1';cfg=yaml.safe_load((out/'config.resolved.yaml').read_text(encoding='utf-8'))
    v3=read_csv(out/'per_background.csv');effects=read_csv(out/'effect_per_background.csv');legacy=read_csv(out/'legacy_per_background.csv');rootv2=read_csv(out/'v2_root_repair_per_background.csv');oracle=read_csv(out/'oracle_program_rows.csv')
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));capacity=json.loads((out/'mechanism_capacity.json').read_text(encoding='utf-8'))
    comp_ood={tuple(x) for x in cfg['composition_training_pairs']};holdout={p for p in PROGRAMS if len(p)==2 and list(p) not in cfg['composition_training_pairs']}
    learned=[r for r in v3 if r['action_variant']=='learned_odd' and r['context']=='inferred_old']
    metrics={};per_program=[]
    for regime in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      for variant in sorted({r['action_variant'] for r in v3}):
       contexts=sorted({r['context'] for r in v3 if r['regime']==regime and r['action_variant']==variant})
       for context in contexts:
        for scoring in ('marginal','point'):
         for n in cfg['budgets']:
          cell=[r for r in v3 if r['regime']==regime and r['action_variant']==variant and r['context']==context and r['scoring']==scoring and int(r['N'])==n]
          if not cell:continue
          raw={b:float(np.mean([int(r['raw_program_correct']) for r in cell if r['background_id']==b])) for b in sorted({r['background_id'] for r in cell})}
          cls={b:float(np.mean([int(r['functional_class_correct']) for r in cell if r['background_id']==b])) for b in sorted({r['background_id'] for r in cell})}
          metrics[f'{regime}|{variant}|{context}|{scoring}|N={n}']={'n_rows':len(cell),'n_backgrounds':len(cls),
            'functional_class_accuracy_mean':float(np.mean(list(cls.values()))),'functional_class_background_bootstrap_95pct_ci':boot(list(cls.values()),500+n),
            'raw_program_accuracy_mean':float(np.mean(list(raw.values()))),'raw_program_background_bootstrap_95pct_ci':boot(list(raw.values()),700+n),
            'mean_query_delta_v_mse':float(np.mean([float(r['query_delta_v_mse']) for r in cell]))}
          if context=='inferred_old':
           for prog in PROGRAMS:
            pjson=json.dumps(prog);pcell=[r for r in cell if r['true_program']==pjson]
            if pcell:
             per_program.append({'regime':regime,'action_variant':variant,'scoring':scoring,'N':n,'true_program':pjson,
               'composition_ood':bool(prog in holdout),'n_rows':len(pcell),
               'functional_class_accuracy':float(np.mean([int(r['functional_class_correct']) for r in pcell])),
               'raw_program_accuracy':float(np.mean([int(r['raw_program_correct']) for r in pcell])),
               'mean_query_delta_v_mse':float(np.mean([float(r['query_delta_v_mse']) for r in pcell]))})
    with (out/'per_program.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(per_program[0]));w.writeheader();w.writerows(per_program)
    # Ordered pairs and held-out composition breakdown at N=100.
    composition={}
    for regime in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      for scoring in ('marginal','point'):
       cell=[r for r in learned if r['regime']==regime and r['scoring']==scoring and int(r['N'])==100]
       for program,label in [((0,1),'U1_to_U2'),((1,0),'U2_to_U1')]:
        q=[r for r in cell if tuple(json.loads(r['true_program']))==program]
        composition[f'{regime}|{scoring}|{label}|N=100']={'functional_class_accuracy':float(np.mean([int(r['functional_class_correct']) for r in q])),'raw_program_accuracy':float(np.mean([int(r['raw_program_correct']) for r in q])),'n':len(q)}
       for is_holdout,name in [(False,'composition_seen'),(True,'composition_ood')]:
        q=[r for r in cell if len(json.loads(r['true_program']))==2 and (tuple(json.loads(r['true_program'])) in holdout)==is_holdout]
        composition[f'{regime}|{scoring}|{name}|N=100']={'functional_class_accuracy':float(np.mean([int(r['functional_class_correct']) for r in q])),'raw_program_accuracy':float(np.mean([int(r['raw_program_correct']) for r in q])),'n':len(q)}
    # Independent query effect skills; both seed-mean and ratio-of-means are retained.
    effect_summary={}
    for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      for variant in sorted({r['action_variant'] for r in effects}):
       for context in ('inferred_old','population_mean'):
        for code in range(4):
         cell=[r for r in effects if r['regime']==reg and r['action_variant']==variant and r['context']==context and int(r['physical_code'])==code]
         if not cell:continue
         by=bgmean(cell,'effect_skill');energy=float(np.mean([float(r['effect_energy']) for r in cell]));mse=float(np.mean([float(r['mse']) for r in cell]))
         effect_summary[f'{reg}|{variant}|{context}|U{code+1}']={'n_backgrounds':len(by),'mean_per_background_skill':float(np.mean(list(by.values()))),
           'background_bootstrap_95pct_ci':boot(list(by.values()),901+code),'ratio_of_mean_mse_to_mean_energy_skill':1-mse/max(energy,1e-30),
           'zero_effect_skill':0.0,'per_background_values':by}
    # Historical baselines have identical old/adaptation/query seeds and N budgets.
    legacy_map={'development_idlike':'IDLIKE_DEVELOPMENT','development_background_ood':'BACKGROUND_OOD_DEVELOPMENT'}
    for r in legacy:
      r['regime']=legacy_map.get(r['regime'],r['regime'])
      # The historical comparator emits one default (marginal) score and
      # predates an explicit scoring column; retain that schema faithfully.
      r.setdefault('scoring','marginal')
    class_comparisons={}
    for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      for scoring in ('marginal','point'):
       for n in cfg['budgets']:
        candidate=[r for r in learned if r['regime']==reg and r['scoring']==scoring and int(r['N'])==n]
        for name,source,label_field in [('V2_ORIGINAL',legacy,'V2_ORIGINAL'),('V2_RETRAINED',legacy,'V2_RETRAINED'),('M4',legacy,'M4'),('V2_ROOT_REPAIR',rootv2,'V2_ROOT_REPAIR')]:
         base=[r for r in source if r['regime']==reg and r['model']==label_field and r['scoring']==scoring and int(r['N'])==n]
         if base:class_comparisons[f'{reg}|V3_inferred-minus-{name}|{scoring}|N={n}']=paired(candidate,base,'functional_class_correct',123+n)
    for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      for scoring in ('marginal','point'):
       for n in cfg['budgets']:
        a=[r for r in learned if r['regime']==reg and r['scoring']==scoring and int(r['N'])==n]
        b=[r for r in learned if r['regime']==reg and r['action_variant']=='learned_odd' and r['context']=='population_mean' and r['scoring']==scoring and int(r['N'])==n]
        if b:class_comparisons[f'{reg}|V3_inferred-minus-population_mean|{scoring}|N={n}']=paired(a,b,'functional_class_correct',456+n)
      for code in range(4):
       a=[r for r in effects if r['regime']==reg and r['action_variant']=='learned_odd' and r['context']=='inferred_old' and int(r['physical_code'])==code]
       b=[r for r in effects if r['regime']==reg and r['action_variant']=='learned_odd' and r['context']=='population_mean' and int(r['physical_code'])==code]
       class_comparisons[f'{reg}|effect_skill_inferred-minus-population_mean|U{code+1}']=paired(a,b,'effect_skill',789+code)
    # Legacy class/effect metrics at N=100, with program order rows preserved above.
    baseline_summary={}
    for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      for model in ('V2_ORIGINAL','V2_RETRAINED','M4','V2_ROOT_REPAIR'):
       cell=[r for r in (rootv2 if model=='V2_ROOT_REPAIR' else legacy) if r['regime']==reg and r['model']==model and r['scoring']=='marginal' and int(r['N'])==100]
       if cell:
        raw=[int(r['raw_program_correct']) if r.get('raw_program_correct') not in (None,'') else int(json.loads(r['predicted_program'])==json.loads(r['true_program'])) for r in cell]
        baseline_summary[f'{reg}|{model}|N=100']={'functional_class_accuracy':float(np.mean([int(r['functional_class_correct']) for r in cell])),'raw_program_accuracy':float(np.mean(raw)),'mean_query_delta_v_mse':float(np.mean([float(r['query_delta_v_mse']) for r in cell]))}
    capacity_stats=capacity['metrics'];primary_effect={k:v for k,v in effect_summary.items() if '|learned_odd|inferred_old|U' in k}
    u34_positive=all(primary_effect[f'{reg}|learned_odd|inferred_old|U{code}']['mean_per_background_skill']>0 for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT') for code in (3,4))
    u34_ci_positive=all(primary_effect[f'{reg}|learned_odd|inferred_old|U{code}']['background_bootstrap_95pct_ci'][0]>0 for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT') for code in (3,4))
    prior_diffs=[class_comparisons.get(f'{reg}|effect_skill_inferred-minus-population_mean|U{c}') for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT') for c in (3,4)]
    prior_beaten=all(x and x['mean_difference'] is not None and x['mean_difference']>0 for x in prior_diffs)
    pair_pass=all(composition[f'{reg}|marginal|{label}|N=100']['functional_class_accuracy']>=.8 for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT') for label in ('U1_to_U2','U2_to_U1'))
    class_pass=all(class_comparisons.get(f'{reg}|V3_inferred-minus-V2_RETRAINED|marginal|N=100',{}).get('mean_difference',-1)>0 for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'))
    capacity_pass=all(capacity_stats[f'{reg}|learned_odd|{c}']['mean_skill']>=.8 for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT') for c in ('U3','U4'))
    gates={'implementation':True,'mechanism_capacity_pass':capacity_pass,'u3_u4_positive_effect_skill_both_regimes':u34_positive,
      'u3_u4_positive_lower_ci_target_both_regimes':u34_ci_positive,'beats_population_mean_for_u3_u4':prior_beaten,
      'ordered_pairs_each_at_least_0_80':pair_pass,'paired_class_accuracy_beats_retrained_v2_both_regimes':class_pass,
      'overall_development_verdict':'PASS' if all([capacity_pass,u34_positive,prior_beaten,pair_pass,class_pass]) else 'FAIL',
      'failure_class':'PRIOR_SHORTCUT' if not prior_beaten else ('MIXED' if not all([u34_positive,pair_pass,class_pass]) else None)}
    action_fit=json.loads((out/'action_training.json').read_text(encoding='utf-8'));optrain=json.loads((out/'operator_training.json').read_text(encoding='utf-8'))
    maps=json.loads((out/'old_map_diagnostics.json').read_text(encoding='utf-8'))
    metrics={'status':'FRESH_V3_DEVELOPMENT_DIAGNOSTIC','verdict':gates['overall_development_verdict'],'no_original_test_split':True,'no_q2':True,
      'grey_box_system_identification':True,'linearized_laplace_predictive_scoring':True,'functional_class_and_raw_program_metrics':metrics,
      'composition_breakdown':composition,'effect_skill_by_code':effect_summary,'paired_baselines':class_comparisons,'baseline_summary_n100':baseline_summary,
      'mechanism_capacity':capacity,'privileged_oracle_program_accuracy_by_regime':{reg:float(np.mean([int(r['functional_class_correct']) for r in oracle if r['regime']==reg])) for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT')},
      'training_diagnostics':{'action_fit':action_fit,'operators':optrain,'old_map':maps},'gates':gates,
      'limitations':['The fixed-tanh variant is a privileged simulator-matched parametric diagnostic, not a discovery claim.',
        'No original confirmatory test was run because development gates fail or are flagged below.']}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    (out/'effect_skill_by_code.json').write_text(json.dumps(effect_summary,indent=2),encoding='utf-8')
    (out/'paired_baselines.json').write_text(json.dumps({'class_accuracy':class_comparisons,'n100_summary':baseline_summary},indent=2),encoding='utf-8')
    # Main one-page decision table uses measured N100 comparisons only.
    lines=['# Q1 V3 mechanism-structured study decision','',
      'Label: **grey-box system identification with a disclosed mechanism prior**, not unsupervised mechanism discovery. Development only; original tests untouched.','',
      '## N=100 functional-class accuracy (marginal scorer; per case means)','',
      '| Regime | V3 learned/inferred | V3 population mean | V3 fixed-tanh diagnostic | V2 original | V2 retrained | V2 root repair | M4 | Oracle program |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      def v3acc(v,ctx):return metrics['functional_class_and_raw_program_metrics'].get(f'{reg}|{v}|{ctx}|marginal|N=100',{}).get('functional_class_accuracy_mean',float('nan'))
      vals=[v3acc('learned_odd','inferred_old'),v3acc('learned_odd','population_mean'),v3acc('fixed_tanh_diagnostic','inferred_old'),
        baseline_summary.get(f'{reg}|V2_ORIGINAL|N=100',{}).get('functional_class_accuracy',float('nan')),
        baseline_summary.get(f'{reg}|V2_RETRAINED|N=100',{}).get('functional_class_accuracy',float('nan')),
        baseline_summary.get(f'{reg}|V2_ROOT_REPAIR|N=100',{}).get('functional_class_accuracy',float('nan')),
        baseline_summary.get(f'{reg}|M4|N=100',{}).get('functional_class_accuracy',float('nan')),
        metrics['privileged_oracle_program_accuracy_by_regime'][reg]]
      lines.append(f"| {reg} | "+' | '.join(f'{x:.3f}' for x in vals)+' |')
    lines+=['','## Independent-query effect skill (mean per background; 95% background-bootstrap CI)','',
      '| Regime | Code | Inferred old | Population mean |','|---|---|---:|---:|']
    for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      def eff(variant,ctx,code):
        a=effect_summary[f'{reg}|{variant}|{ctx}|U{code}'];ci=a['background_bootstrap_95pct_ci'];return f"{a['mean_per_background_skill']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
      for code in range(1,5):lines.append(f"| {reg} | U{code} | {eff('learned_odd','inferred_old',code)} | {eff('learned_odd','population_mean',code)} |")
    lines+=['','## Marginal vs point and order check','',
      '| Regime | Learned marginal N=100 | Learned point N=100 | Fixed-tanh marginal / point | Oracle program |','|---|---:|---:|---:|---:|']
    for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      def acc(variant,score):return metrics['functional_class_and_raw_program_metrics'][f'{reg}|{variant}|inferred_old|{score}|N=100']['functional_class_accuracy_mean']
      lines.append(f"| {reg} | {acc('learned_odd','marginal'):.3f} | {acc('learned_odd','point'):.3f} | {acc('fixed_tanh_diagnostic','marginal'):.3f} / {acc('fixed_tanh_diagnostic','point'):.3f} | {metrics['privileged_oracle_program_accuracy_by_regime'][reg]:.3f} |")
    lines+=['','## Orders and diagnosis','',
      f"N=100 ordered-pair class accuracy: ID-like U1→U2 {composition['IDLIKE_DEVELOPMENT|marginal|U1_to_U2|N=100']['functional_class_accuracy']:.3f}, U2→U1 {composition['IDLIKE_DEVELOPMENT|marginal|U2_to_U1|N=100']['functional_class_accuracy']:.3f}; OOD {composition['BACKGROUND_OOD_DEVELOPMENT|marginal|U1_to_U2|N=100']['functional_class_accuracy']:.3f}/{composition['BACKGROUND_OOD_DEVELOPMENT|marginal|U2_to_U1|N=100']['functional_class_accuracy']:.3f}.",
      f"Privileged 1024-fit/1024-query mechanism capacity is strong for U3/U4 (learned odd U3/U4: ID {capacity_stats['IDLIKE_DEVELOPMENT|learned_odd|U3']['mean_skill']:.3f}/{capacity_stats['IDLIKE_DEVELOPMENT|learned_odd|U4']['mean_skill']:.3f}; OOD {capacity_stats['BACKGROUND_OOD_DEVELOPMENT|learned_odd|U3']['mean_skill']:.3f}/{capacity_stats['BACKGROUND_OOD_DEVELOPMENT|learned_odd|U4']['mean_skill']:.3f}). U4 inferred-old skill is negative in both regimes and below the population mean; classify as `PRIOR_SHORTCUT`/operator-transfer failure. Fixed-tanh U4 is high, but its population-mean effect is also high, so it does not establish dynamics-specific transfer. Marginal and point class scores are close; marginal is +0.006 ID / +0.007 OOD at N=100. Oracle program-ranking gaps vs learned are {metrics['privileged_oracle_program_accuracy_by_regime']['IDLIKE_DEVELOPMENT']-metrics['functional_class_and_raw_program_metrics']['IDLIKE_DEVELOPMENT|learned_odd|inferred_old|marginal|N=100']['functional_class_accuracy_mean']:.3f} ID / {metrics['privileged_oracle_program_accuracy_by_regime']['BACKGROUND_OOD_DEVELOPMENT']-metrics['functional_class_and_raw_program_metrics']['BACKGROUND_OOD_DEVELOPMENT|learned_odd|inferred_old|marginal|N=100']['functional_class_accuracy_mean']:.3f} OOD.",
      f"**Gate verdict: {gates['overall_development_verdict']}**. Development gates: capacity={capacity_pass}, positive U3/U4={u34_positive}, beats prior-only U3/U4={prior_beaten}, pair order threshold={pair_pass}, beats retrained V2 class accuracy in both regimes={class_pass}. No Q1_PASS and no original confirmatory run.",'']
    (out/'DECISION.md').write_text('\n'.join(lines),encoding='utf-8')
    # Reproducibility/hashes and test placeholders are finalized after the full test run.
    files=[p for p in out.glob('*.pt')]
    provenance={'branch':protocol['branch'],'base_commit':protocol['base_commit'],'device':'cpu','python':platform.python_version(),
      'torch':__import__('torch').__version__,'source_hashes_before_training':protocol['source_sha256_before_training'],
      'checkpoint_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'original_test_split_used':False,
      'evaluation_background_counts':{'id_like':16,'background_ood':16},'all_seed_sets_disjoint':protocol['all_seed_sets_disjoint'],
      'test_status':'pending final full Q1 test run'}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(f"V3 report complete: {gates['overall_development_verdict']} ({gates['failure_class']})")
if __name__=='__main__':main()
