"""Aggregate the saved, paired Q1 V2 operator retraining development run."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np, torch, yaml
from q1.oracle import functional_classes
from q1.simulator import DynamicsParams, PROGRAMS

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read_csv(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='results/q1/q1_v2_operator_retrain_dev1');args=ap.parse_args()
    out=ROOT/args.run_dir;cfg=yaml.safe_load((out/'config.resolved.yaml').read_text(encoding='utf-8'))
    manifest=json.loads((out/'split_manifest.json').read_text(encoding='utf-8'))
    rows=read_csv(out/'per_background.csv')
    for row in rows:
        for k in ('N','functional_class_correct','optimizer_seed'):
            row[k]=int(row[k])
        row['true_program']=tuple(json.loads(row['true_program']))
        row['predicted_program']=tuple(json.loads(row['predicted_program']))
        row['point_predicted_program']=tuple(json.loads(row['point_predicted_program']))
    class_by_bg={}
    for regime in ('development_idlike','development_background_ood'):
        for bi,bg in enumerate(manifest[regime]['backgrounds']):
            params=DynamicsParams(np.asarray(bg['B']),float(bg['d']),float(bg['b']),float(bg['kappa']))
            class_by_bg[f"{'IDLIKE' if regime=='development_idlike' else 'BACKGROUND_OOD'}_DEVELOPMENT-{bi:03d}"]=functional_classes(params,dt=cfg['dt'])[0]
    metrics={}
    for regime in sorted({r['regime'] for r in rows}):
        for n in cfg['budgets']:
            cell=[r for r in rows if r['regime']==regime and r['N']==n]
            point_correct=[]
            for r in cell:
                classes=class_by_bg[r['background_id']]
                point_correct.append(int(classes[PROGRAMS.index(r['point_predicted_program'])]==
                                         classes[PROGRAMS.index(r['true_program'])]))
            metrics[f'{regime}|N={n}']={
                'n_background_program_seed_rows':len(cell),
                'functional_class_accuracy_bayesian_marginal':float(np.mean([r['functional_class_correct'] for r in cell])),
                'functional_class_accuracy_point_estimate':float(np.mean(point_correct)),
                'mean_query_delta_v_mse':float(np.mean([float(r['query_delta_v_mse']) for r in cell])),
            }
    effects={}
    for i in range(4):
        selected=[r for r in rows if r['N']==100 and r['true_program']==(i,) and r['effect_skill']!='']
        skills=[float(r['effect_skill']) for r in selected]
        effects[f'U{i+1}']={'n_background_seed_rows':len(selected),'mean_effect_skill':float(np.mean(skills)) if skills else None,
                            'median_effect_skill':float(np.median(skills)) if skills else None}
        if i==3:
            analytic=[float(r['analytical_u4_constant_drag_skill']) for r in selected if r['analytical_u4_constant_drag_skill']!='']
            effects['U4']['analytic_constant_drag_reference_mean_skill']=float(np.mean(analytic)) if analytic else None
    compositions={}
    for pair,label in [((0,1),'(U1,U2)'),((1,0),'(U2,U1)')]:
        selected=[r for r in rows if r['N']==100 and r['true_program']==pair]
        compositions[label]={'n_background_seed_rows':len(selected),
            'functional_class_accuracy':float(np.mean([r['functional_class_correct'] for r in selected]))}
    comparison={}
    for regime,metric_regime in [('IDLIKE_DEVELOPMENT','IDLIKE_DEVELOPMENT'),
                                 ('BACKGROUND_OOD_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT')]:
        for n in cfg['budgets']:
            cell=[r for r in rows if r['regime']==regime and r['N']==n]
            comparison[f'{regime}|V2_RETRAINED|N={n}']={
                'n_background_program_seed_rows':len(cell),
                'functional_class_accuracy':float(np.mean([int(r['functional_class_correct']) for r in cell])),
                'mean_query_delta_v_mse':float(np.mean([float(r['query_delta_v_mse']) for r in cell]))}
    legacy_path=out/'legacy_per_background.csv'
    if legacy_path.exists():
        legacy=read_csv(legacy_path)
        for regime in sorted({r['regime'] for r in legacy}):
            for model in ('V2_ORIGINAL','M4'):
                for n in cfg['budgets']:
                    cell=[r for r in legacy if r['regime']==regime and r['model']==model and int(r['N'])==n]
                    if cell:
                        comparison[f'{regime}|{model}|N={n}']={
                            'n_background_program_cases':len(cell),
                            'functional_class_accuracy':float(np.mean([int(r['functional_class_correct']) for r in cell])),
                            'mean_query_delta_v_mse':float(np.mean([float(r['query_delta_v_mse']) for r in cell]))}
    train=json.loads((out/'operator_training.json').read_text(encoding='utf-8'))
    coverage_ok=all(e['training_episode_count']==480 and e['unique_training_episodes']==480
        for seed in ('4201','4202') for e in train['curves'][seed]['epoch_coverage'])
    result={'status':'FRESH_DEVELOPMENT_DIAGNOSTIC','basis_reused_unchanged':True,
      'training':{'full_coverage_every_epoch':coverage_ok,'training_episodes_per_epoch':480,
        'optimizer_seeds':cfg['optimizer_seeds'],'epochs':cfg['operator_epochs'],
        'optimizer_updates_per_final_bank':480*int(cfg['operator_epochs']),
        'original_v2_optimizer_updates_per_bank':160,
        'selected_regularization':train['selected_regularization'],
        'selected_best_epoch_by_seed':{s:train['curves'][s]['best_epoch'] for s in ('4201','4202')},
        'selection_split':'inner-training backgrounds only'},
      'metrics':metrics,'single_intervention_effect_skill_N100':effects,
      'ordered_composition_N100':compositions,'paired_same_fresh_background_comparison':comparison,
      'test_split_used':False}
    (out/'metrics_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'diagnosis.md').write_text(make_diagnosis(result),encoding='utf-8')
    try:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for seed in ('4201','4202'):
            curve=train['curves'][seed]['curve'];xs=[x['epoch'] for x in curve]
            axes[0].plot(xs,[x['train_nll'] for x in curve],marker='o',label=f'seed {seed}')
            axes[1].plot(xs,[x['inner_nll'] for x in curve],marker='o',label=f'seed {seed}')
        axes[0].set(title='Operator training NLL',xlabel='Full epoch',ylabel='Mean training NLL')
        axes[1].set(title='Inner-training checkpoint selection',xlabel='Full epoch',ylabel='Inner predictive NLL')
        for ax in axes:ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();fig.savefig(out/'operator_training_curves.png',dpi=160);plt.close(fig)
    except ImportError:
        pass
    if not (out/'metrics.json').exists():
        by_cell={}
        for r in rows:by_cell.setdefault(f"{r['regime']}|N={r['N']}",[]).append(r)
        metrics_json={k:{'n_background_program_seed_rows':len(v),
          'functional_class_accuracy':float(np.mean([int(x['functional_class_correct']) for x in v])),
          'mean_query_delta_v_mse':float(np.mean([float(x['query_delta_v_mse']) for x in v])),
          'mean_true_vs_best_wrong_class_margin':float(np.mean([float(x['true_vs_best_wrong_class_margin']) for x in v]))}
          for k,v in by_cell.items()}
        (out/'metrics.json').write_text(json.dumps({'status':'DEVELOPMENT_DIAGNOSTIC','length_penalty':json.loads((out/'length_penalty_selection.json').read_text())['selected'],
          'metrics':metrics_json,'limitations':['Fresh development diagnostic only; no original test split used.','Assessment query labels are used only after selection for query-MSE reporting.']},indent=2),encoding='utf-8')
        pre=json.loads((out/'provenance.pretraining.json').read_text(encoding='utf-8'))
        source_paths=[ROOT/'q1_v2/basis.py',ROOT/'q1_v2/bayes.py',ROOT/'q1_v2/operators.py',ROOT/'q1_v2/inference.py',ROOT/'q1_v2/training.py',ROOT/'scripts/q1_v2_pipeline.py',ROOT/'scripts/q1_v2_legacy_compare.py',ROOT/'scripts/q1_v2_operator_retrain_summary.py',ROOT/'configs/q1_v2_operator_retrain.yaml']
        prov={**pre,'runtime_seconds':None,'runtime_note':'Pipeline evaluation completed; final aggregation was resumed from saved per_background.csv after correcting a key mismatch.',
          'source_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
          'checkpoint_hashes':{p.name:sha(p) for p in out.glob('*.pt')},
          'optimizer_seeds':cfg['optimizer_seeds'],'results_rows':len(rows)}
        (out/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
    print(f'Wrote retraining summaries under {out}')

def make_diagnosis(r):
    rows=['# Q1 V2 operator retraining diagnosis','',
      'This is a fresh development diagnostic. The original V2 basis/prior checkpoint and architecture were reused unchanged; the operator banks were trained on all 480 training episodes in each epoch, with balanced opaque-code ordering. Operator checkpoint epochs were selected by predictive NLL on inner-training backgrounds. No test split was used.','',
      '## Retrained operator results','',
      '| Regime | N | Bayesian marginal | Point estimate | Rows |','|---|---:|---:|---:|---:|']
    for k,v in r['metrics'].items():rows.append(f"| {k.split('|')[0]} | {k.split('N=')[1]} | {v['functional_class_accuracy_bayesian_marginal']:.3f} | {v['functional_class_accuracy_point_estimate']:.3f} | {v['n_background_program_seed_rows']} |")
    rows+=['','## Single-intervention effect skill at N=100','','| Intervention | Mean skill | Median skill | Rows |','|---|---:|---:|---:|']
    for k,v in r['single_intervention_effect_skill_N100'].items():rows.append(f"| {k} | {v['mean_effect_skill'] if v['mean_effect_skill'] is not None else float('nan'):.3f} | {v['median_effect_skill'] if v['median_effect_skill'] is not None else float('nan'):.3f} | {v['n_background_seed_rows']} |")
    rows+=['','## Ordered compositions at N=100','','| Program | Functional-class accuracy | Rows |','|---|---:|---:|']
    for k,v in r['ordered_composition_N100'].items():rows.append(f"| {k} | {v['functional_class_accuracy']:.3f} | {v['n_background_seed_rows']} |")
    rows+=['','## Comparison on the same fresh development cases','','| Regime / model / N | Functional-class accuracy | Mean query Δv MSE | Cases |','|---|---:|---:|---:|']
    for k,v in r['paired_same_fresh_background_comparison'].items():rows.append(f"| {k} | {v['functional_class_accuracy']:.3f} | {v['mean_query_delta_v_mse']:.3g} | {v.get('n_background_program_cases',v.get('n_background_program_seed_rows'))} |")
    rows+=['','## Diagnosis','',
      f"At N=100 the fresh ID-like accuracy rose from {r['paired_same_fresh_background_comparison']['development_idlike|V2_ORIGINAL|N=100']['functional_class_accuracy']:.3f} for the original banks to {r['paired_same_fresh_background_comparison']['IDLIKE_DEVELOPMENT|V2_RETRAINED|N=100']['functional_class_accuracy']:.3f} after full-coverage retraining. On background-OOD it changed from {r['paired_same_fresh_background_comparison']['development_background_ood|V2_ORIGINAL|N=100']['functional_class_accuracy']:.3f} to {r['paired_same_fresh_background_comparison']['BACKGROUND_OOD_DEVELOPMENT|V2_RETRAINED|N=100']['functional_class_accuracy']:.3f}. Each retrained bank received 1,920 updates versus 160 in the old loop, so this experiment tests better coverage together with a larger training budget and cannot attribute the gain to coverage alone. The extreme negative U4 effect skill ({r['single_intervention_effect_skill_N100']['U4']['mean_effect_skill']:.1f}) and negative U3 skill ({r['single_intervention_effect_skill_N100']['U3']['mean_effect_skill']:.2f}) show that additional training did not resolve all operator failures. Ordered (U1,U2) and (U2,U1) classes score {r['ordered_composition_N100']['(U1,U2)']['functional_class_accuracy']:.3f} and {r['ordered_composition_N100']['(U2,U1)']['functional_class_accuracy']:.3f}. This is development evidence only, not acceptance or test-set confirmation.",'']
    return '\n'.join(rows)

if __name__=='__main__':main()
