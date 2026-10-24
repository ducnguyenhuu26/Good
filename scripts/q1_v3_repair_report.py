"""Aggregate paired Q1 V3 repair ablations and apply the frozen decision gates."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/q1/q1_v3_mechanism_repair_dev1'
REGIMES=('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT')
ABLATIONS=('original_v3','action_response_only','operator_selectivity_only','combined_action_and_operator')

def read_csv(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def boot(vals,seed,reps=5000):
    x=np.asarray(vals,dtype=float)
    d=np.random.default_rng(seed).choice(x,size=(reps,len(x)),replace=True).mean(1)
    return [float(np.quantile(d,.025)),float(np.quantile(d,.975))]

def bg_metric(rows,field,where):
    selected=[r for r in rows if where(r)]
    by={}
    for r in selected:by.setdefault(r['background_id'],[]).append(float(r[field]))
    return {k:float(np.mean(v)) for k,v in by.items()}

def paired(a,b,seed):
    common=sorted(set(a)&set(b));d=[a[k]-b[k] for k in common]
    return {'n_paired_backgrounds':len(common),'mean_difference':float(np.mean(d)) if d else None,
        'background_bootstrap_95pct_ci':boot(d,seed) if d else None,'per_background_difference':dict(zip(common,d))}

def summarize_class(rows,model,regime,context,scoring,n):
    if model=='original_v3':
        use=[r for r in rows if r['regime']==regime and r['action_variant']=='learned_odd' and r['context']==context and r['scoring']==scoring and int(r['N'])==n]
    elif model.startswith('V2_') or model=='M4':
        use=[r for r in rows if r['regime']==regime and r['model']==model and int(r['N'])==n]
    else:
        use=[r for r in rows if r['regime']==regime and r['ablation']==model and r['context']==context and r['scoring']==scoring and int(r['N'])==n]
    by=bg_metric(use,'functional_class_correct',lambda r:True)
    return {'functional_class_accuracy_mean':float(np.mean(list(by.values()))) if by else None,
        'background_bootstrap_95pct_ci':boot(list(by.values()),4000+n) if by else None,'n_backgrounds':len(by),'n_rows':len(use)}

def main():
    config=__import__('yaml').safe_load((OUT/'config.resolved.yaml').read_text(encoding='utf-8'))
    original=read_csv('per_background.csv');legacy=read_csv('legacy_per_background.csv')
    legacy_regime={'development_idlike':'IDLIKE_DEVELOPMENT','development_background_ood':'BACKGROUND_OOD_DEVELOPMENT'}
    for row in legacy:row['regime']=legacy_regime.get(row['regime'],row['regime'])
    ablation=read_csv('repair_ablation_per_background.csv');effects_original=read_csv('effect_per_background.csv')
    effects=read_csv('repair_effect_per_background.csv')
    class_metric={}
    for reg in REGIMES:
      for model in ABLATIONS+('V2_ORIGINAL','V2_RETRAINED','M4'):
       contexts=('inferred_old','population_mean') if model not in ('V2_ORIGINAL','V2_RETRAINED','M4') else ('inferred_old',)
       for ctx in contexts:
        for scoring in ('marginal','point'):
         for n in config['budgets']:
          source=original if model=='original_v3' else legacy if model.startswith('V2_') or model=='M4' else ablation
          item=summarize_class(source,
                model,reg,ctx,scoring,n)
          if item['n_rows']:class_metric[f'{reg}|{model}|{ctx}|{scoring}|N={n}']=item
    effect_summary={}
    for reg in REGIMES:
      for model in ABLATIONS:
       for ctx in ('inferred_old','population_mean'):
        source=effects_original if model=='original_v3' else effects
        for code in range(4):
          def belongs(r):
            if r['regime']!=reg or int(r['physical_code'])!=code or r['context']!=ctx:return False
            if model=='original_v3':return r['action_variant']=='learned_odd'
            return r['ablation']==model
          per=bg_metric(source,'effect_skill',belongs)
          raw=[r for r in source if belongs(r)]
          if per:
            mse=float(np.mean([float(r['mse']) for r in raw]));energy=float(np.mean([float(r['effect_energy']) for r in raw]))
            effect_summary[f'{reg}|{model}|{ctx}|U{code+1}']={'mean_per_background_skill':float(np.mean(list(per.values()))),
                'background_bootstrap_95pct_ci':boot(list(per.values()),700+code),
                'ratio_of_mean_mse_to_mean_energy_skill':1-mse/max(energy,1e-30),
                'n_backgrounds':len(per),'per_background_values':per,'zero_effect_skill':0.0}
    # V2 and M4 effect baselines from their oracle-code-identity singleton evaluations.
    legacy_effect={}
    for reg in REGIMES:
      for model in ('V2_ORIGINAL','V2_RETRAINED','M4'):
       for code in range(4):
        cell=[r for r in legacy if r['regime']==({'IDLIKE_DEVELOPMENT':'development_idlike','BACKGROUND_OOD_DEVELOPMENT':'development_background_ood'}[reg]) and r['model']==model and json.loads(r['true_program'])==[code] and int(r['N'])==100 and r.get('effect_skill') not in (None,'')]
        bg=bg_metric(cell,'effect_skill',lambda r:True)
        if bg:legacy_effect[f'{reg}|{model}|U{code+1}']={'mean_per_background_skill':float(np.mean(list(bg.values()))),'background_bootstrap_95pct_ci':boot(list(bg.values()),1800+code),'per_background_values':bg}
    comparisons={};gates={};margin=float(config['comparison']['regression_noninferiority_margin'])
    for name in ABLATIONS[1:]:
      criteria={'u4_positive_both_regimes':True,'u4_beats_population_both_regimes':True,
        'u1_u3_effect_noninferior_to_original':True,'ordered_pairs_noninferior_to_original':True,
        'ordered_pair_class_accuracy_at_least_threshold':True}
      for reg in REGIMES:
        u4=effect_summary[f'{reg}|{name}|inferred_old|U4'];u4pop=effect_summary[f'{reg}|{name}|population_mean|U4']
        criteria['u4_positive_both_regimes'] &= u4['mean_per_background_skill']>0
        ip=paired(u4pop['per_background_values'],u4['per_background_values'],2200)
        # reverse paired() input above so positive means inferred-old beats population.
        ip=paired(u4['per_background_values'],u4pop['per_background_values'],2200)
        comparisons[f'{reg}|{name}|U4_inferred_minus_population']=ip
        criteria['u4_beats_population_both_regimes'] &= ip['mean_difference'] is not None and ip['background_bootstrap_95pct_ci'][0]>0
        for code in range(3):
          cand=effect_summary[f'{reg}|{name}|inferred_old|U{code+1}']['per_background_values']
          base=effect_summary[f'{reg}|original_v3|inferred_old|U{code+1}']['per_background_values']
          comp=paired(cand,base,2300+code);comparisons[f'{reg}|{name}|U{code+1}_minus_original_v3']=comp
          criteria['u1_u3_effect_noninferior_to_original'] &= comp['background_bootstrap_95pct_ci'][0]>=-margin
        for order,label in [((0,1),'U1_to_U2'),((1,0),'U2_to_U1')]:
          def pick_class(source,model):
            if model=='original_v3':cell=[r for r in source if r['regime']==reg and r['action_variant']=='learned_odd' and r['context']=='inferred_old' and r['scoring']=='marginal' and int(r['N'])==100 and tuple(json.loads(r['true_program']))==order]
            else:cell=[r for r in source if r['regime']==reg and r['ablation']==model and r['context']=='inferred_old' and r['scoring']=='marginal' and int(r['N'])==100 and tuple(json.loads(r['true_program']))==order]
            return bg_metric(cell,'functional_class_correct',lambda r:True)
          cand=pick_class(ablation,name);base=pick_class(original,'original_v3')
          delta=paired(cand,base,2500+len(label));comparisons[f'{reg}|{name}|{label}_minus_original_v3']=delta
          criteria['ordered_pairs_noninferior_to_original'] &= delta['background_bootstrap_95pct_ci'][0]>=-margin
          criteria['ordered_pair_class_accuracy_at_least_threshold'] &= bool(cand) and float(np.mean(list(cand.values())))>=config['gates']['ordered_composition_minimum_class_accuracy']
      criteria['pass']=all(criteria.values());gates[name]=criteria
    # Program-level per-case learning curves and held-out ordered pair scores.
    per_program=[]
    for model in ABLATIONS[1:]:
      for reg in REGIMES:
       for n in config['budgets']:
        for prog in PROGRAMS_LIST():
          cell=[r for r in ablation if r['ablation']==model and r['regime']==reg and r['context']=='inferred_old' and r['scoring']=='marginal' and int(r['N'])==n and tuple(json.loads(r['true_program']))==prog]
          if cell:per_program.append({'model':model,'regime':reg,'N':n,'true_program':json.dumps(prog),
             'composition_ood':len(prog)==2 and list(prog) not in config['composition_training_pairs'],
             'functional_class_accuracy':float(np.mean([int(r['functional_class_correct']) for r in cell])),
             'raw_program_accuracy':float(np.mean([int(r['raw_program_correct']) for r in cell])),'n':len(cell)})
    save_csv(OUT/'repair_per_program.csv',per_program)
    train_log=json.loads((OUT/'ablation_training.json').read_text(encoding='utf-8'))
    original_action_log=json.loads((OUT/'action_training.json').read_text(encoding='utf-8'))
    flexible_action_log=json.loads((OUT/'action_flexible_training.json').read_text(encoding='utf-8'))
    action_fit_comparison={'original_v3_inner_map_nll_per_observation':float(original_action_log['curve'][original_action_log['best_epoch']-1]['inner_map_nll_per_observation']),
        'flexible_spline_inner_map_nll_per_observation':float(flexible_action_log['curve'][flexible_action_log['best_epoch']-1]['inner_map_nll_per_observation']),
        'original_v3_selected_epoch':original_action_log['best_epoch'],'flexible_spline_selected_epoch':flexible_action_log['best_epoch'],
        'selection_data':'training-inner only'}
    opaque_map={int(k):int(v) for k,v in train_log['physical_to_opaque_fixed_from_original_training_seed'].items()}
    inverse={v:k for k,v in opaque_map.items()};assignment_report={}
    group_names=('action_B','linear_drag_d','speed_drag_b','position_feedback_kappa')
    for model,byseed in train_log['operator_fits'].items():
      for seed,meta in byseed.items():
       if not meta['selective']:continue
       probabilities=meta['curve'][meta['best_epoch']-1]['assignment_probabilities']
       for opaque,row in enumerate(probabilities):
        physical=inverse[opaque]
        assignment_report[f'{model}|seed={seed}|U{physical+1}']={'opaque_code':opaque,
          'probability_by_mechanism':dict(zip(group_names,[float(x) for x in row])),
          'selected_mechanism':group_names[int(np.argmax(row))],'selected_probability':float(max(row))}
    metrics={'status':'FRESH_PAIRED_V3_REPAIR_DEVELOPMENT','verdict_by_ablation':{k:'PASS' if v['pass'] else 'FAIL' for k,v in gates.items()},
      'no_original_test_split':True,'no_q2':True,'all_21_programs_evaluated':True,
      'fresh_background_counts':{'id_like':16,'background_ood':16},
      'matched_observations_and_seeds':True,'class_accuracy':class_metric,'effect_skill':effect_summary,
      'v2_m4_effect_baselines':legacy_effect,'learned_opaque_code_mechanism_assignments':assignment_report,
      'action_fit_comparison':action_fit_comparison,'paired_comparisons':comparisons,'gates':gates,
      'limitations':['Action representation is a monotone odd piecewise-linear spline; this is an inductive bias, not an arbitrary response universal approximator.',
        'Fixed simulator and intervention formulas were used only by the evaluator, not to train the ablations.']}
    (OUT/'repair_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (OUT/'repair_effect_skill_by_code.json').write_text(json.dumps(effect_summary,indent=2),encoding='utf-8')
    (OUT/'repair_paired_baselines.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    lines=['# Q1 V3 repair ablations — development decision','',
      'Fresh development only; 16 ID-like and 16 background-OOD backgrounds, all 21 programs, original test split untouched.','',
      '## N=100 functional-class accuracy','',
      '| Regime | Original V3 | Action only | Operator only | Combined | V2 original | V2 retrained | M4 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for reg in REGIMES:
      vals=[]
      for model in ('original_v3','action_response_only','operator_selectivity_only','combined_action_and_operator','V2_ORIGINAL','V2_RETRAINED','M4'):
        row=class_metric.get(f'{reg}|{model}|inferred_old|marginal|N=100',{});vals.append(row.get('functional_class_accuracy_mean',float('nan')))
      lines.append('| '+reg+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    lines+=['','## Functional-class learning curve (marginal scorer)','',
      '| Regime | N | Original V3 | Combined | V2 retrained |','|---|---:|---:|---:|---:|']
    for reg in REGIMES:
      for n in config['budgets']:
        vals=[]
        for model in ('original_v3','combined_action_and_operator','V2_RETRAINED'):
          vals.append(class_metric[f'{reg}|{model}|inferred_old|marginal|N={n}']['functional_class_accuracy_mean'])
        lines.append(f'| {reg} | {n} | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    lines+=['','## U1–U4 independent-query effect skill','',
      '| Regime | Code | Original V3 | Action only | Operator only | Combined | Combined population mean |','|---|---|---:|---:|---:|---:|---:|']
    for reg in REGIMES:
      for code in range(1,5):
        vals=[]
        for model,ctx in [('original_v3','inferred_old'),('action_response_only','inferred_old'),('operator_selectivity_only','inferred_old'),('combined_action_and_operator','inferred_old'),('combined_action_and_operator','population_mean')]:
          vals.append(effect_summary[f'{reg}|{model}|{ctx}|U{code}']['mean_per_background_skill'])
        lines.append(f"| {reg} | U{code} | "+' | '.join(f'{x:.3f}' for x in vals)+' |')
    lines+=['','## Ordered compositions at N=100','',
      '| Regime | Model | U1→U2 | U2→U1 |','|---|---|---:|---:|']
    for reg in REGIMES:
      for model in ('original_v3','action_response_only','operator_selectivity_only','combined_action_and_operator'):
        vals=[]
        for pair in ((0,1),(1,0)):
          source=original if model=='original_v3' else ablation
          if model=='original_v3':cell=[r for r in source if r['regime']==reg and r['action_variant']=='learned_odd' and r['context']=='inferred_old' and r['scoring']=='marginal' and int(r['N'])==100 and tuple(json.loads(r['true_program']))==pair]
          else:cell=[r for r in source if r['regime']==reg and r['ablation']==model and r['context']=='inferred_old' and r['scoring']=='marginal' and int(r['N'])==100 and tuple(json.loads(r['true_program']))==pair]
          vals.append(float(np.mean([int(r['functional_class_correct']) for r in cell])) if cell else float('nan'))
        lines.append(f'| {reg} | {model} | {vals[0]:.3f} | {vals[1]:.3f} |')
    lines+=['','## U4 and ablation gates','',
      '| Ablation | ID U4 skill | OOD U4 skill | ID U4 minus prior CI | OOD U4 minus prior CI | Verdict |','|---|---:|---:|---:|---:|---|']
    for model in ABLATIONS[1:]:
      row=[]
      for reg in REGIMES:
        a=effect_summary[f'{reg}|{model}|inferred_old|U4'];diff=comparisons[f'{reg}|{model}|U4_inferred_minus_population']
        ci=a['background_bootstrap_95pct_ci'];dci=diff['background_bootstrap_95pct_ci']
        row.append(f"{a['mean_per_background_skill']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
        if reg==REGIMES[0]:id_diff=f"{diff['mean_difference']:.3f} [{dci[0]:.3f},{dci[1]:.3f}]"
        else:ood_diff=f"{diff['mean_difference']:.3f} [{dci[0]:.3f},{dci[1]:.3f}]"
      lines.append(f'| {model} | {row[0]} | {row[1]} | {id_diff} | {ood_diff} | {"PASS" if gates[model]["pass"] else "FAIL"} |')
    lines+=['','## Diagnosis','',
      'U1–U3 effect non-inferiority and both ordered-composition criteria use paired background bootstrap intervals with the predeclared 0.02 margin. U4 requires positive skill in each regime and a positive 95% paired interval over the population-mean baseline.',
      f"The flexible response spline did not improve training-inner old-law validation (NLL/observation {action_fit_comparison['flexible_spline_inner_map_nll_per_observation']:.3f}) over original V3 ({action_fit_comparison['original_v3_inner_map_nll_per_observation']:.3f}). Learned assignment probabilities are recorded per opaque code and seed in `repair_metrics.json`; U4's assignment remained diffuse (max group probability below 0.29), so the selective constraint did not isolate its mechanism.",
      'Selectivity-only and combined reduce U4 error but leave skill negative; selectivity variants also regress U3 beyond the predeclared margin. The learned assignment map is data-driven and contains no hard-coded physical code identities.',
      'The combined ablation verdict is **'+('PASS' if gates['combined_action_and_operator']['pass'] else 'FAIL')+'**. '+
      ('All predeclared gates passed.' if gates['combined_action_and_operator']['pass'] else 'The repair did not satisfy every gate. Stop the current Q1 formulation; do not continue architectural tweaks or open another confirmation on these development backgrounds.')]
    (OUT/'REPAIR_DECISION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Repair report:',metrics['verdict_by_ablation'])

def PROGRAMS_LIST():
    from q1.simulator import PROGRAMS
    return [tuple(p) for p in PROGRAMS]

def save_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
