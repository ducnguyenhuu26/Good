"""Summarize cached v2 score margins, controls, and background-paired intervals."""
from __future__ import annotations
import csv,json,sys,hashlib,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import t as student_t
import torch
from q1.simulator import PROGRAMS,DynamicsParams
from q1.oracle import functional_classes
from q1_v2.operators import AffineOperatorBank

def ci(x):
    a=np.asarray(x,float); n=len(a)
    if n<2:return [float(a.mean()),float(a.mean())]
    return [float(a.mean()-student_t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n)),float(a.mean()+student_t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n))]
def pg(s):return tuple(json.loads(s))
def main():
    import argparse,yaml
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='results/q1/q1_v2_bayesian_operators_dev1');a=p.parse_args();out=ROOT/a.run_dir
    cfg=yaml.safe_load((out/'config.resolved.yaml').read_text(encoding='utf-8'));manifest=json.loads((out/'split_manifest.json').read_text(encoding='utf-8'))
    classes={}
    for reg in ('development_idlike','development_background_ood'):
        label='IDLIKE_DEVELOPMENT' if reg=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
        for i,bg in enumerate(manifest[reg]['backgrounds']):
            par=DynamicsParams(np.asarray(bg['B']),bg['d'],bg['b'],bg['kappa'])
            classes[f'{label}-{i:03d}']=functional_classes(par,dt=cfg['dt'])[0]
    records=[]
    for r in csv.DictReader((out/'per_background.csv').open(encoding='utf-8')):
        cls=classes[r['background_id']]; truth=cls[PROGRAMS.index(pg(r['true_program']))]
        for column,model in [('predicted_program','marginal_correct'),('point_predicted_program','point_correct'),
            ('single_only_predicted_program','single_only_correct'),('wrong_code_predicted_program','wrong_code_correct'),
            ('context_ablated_predicted_program','context_ablated_correct')]:
            pred=pg(r[column]);records.append({'regime':r['regime'],'N':r['N'],'background_id':r['background_id'],
                'optimizer_seed':r['optimizer_seed'],'model':model,'correct':int(cls[PROGRAMS.index(pred)]==truth)})
        records.append({'regime':r['regime'],'N':r['N'],'background_id':r['background_id'],
            'optimizer_seed':r['optimizer_seed'],'model':'identity_only_correct','correct':int(cls[0]==truth)})
    for r in csv.DictReader((out/'legacy_per_background.csv').open(encoding='utf-8')):
        reg='IDLIKE_DEVELOPMENT' if r['regime']=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
        bi=int(r['background_id'].rsplit('-',1)[1]); bgid=f'{reg}-{bi:03d}'
        records.append({'regime':reg,'N':r['N'],'background_id':bgid,
            'optimizer_seed':'legacy','model':r['model']+'_correct','correct':int(r['functional_class_correct'])})
    grouped={}
    for r in records:grouped.setdefault((r['regime'],str(r['N']),r['background_id'],r['model']),[]).append(r['correct'])
    bgmodel={}
    for (reg,n,bg,model),vals in grouped.items():bgmodel.setdefault((reg,n,model),{})[bg]=float(np.mean(vals))
    cells={}
    for (reg,n,model),bybg in bgmodel.items():
        vals=list(bybg.values());cells.setdefault(f'{reg}|N={n}',{})[model]={'mean':float(np.mean(vals)),
            'ci95_background':ci(vals),'n_backgrounds':len(vals)}
    paired={}
    for key,models in cells.items():
        reg,n=key.rsplit('|N=',1);v2=bgmodel[(reg,n,'marginal_correct')]
        paired[key]={}
        for (r2,n2,model),bybg in bgmodel.items():
            if r2!=reg or n2!=n or model=='marginal_correct':continue
            common=sorted(set(v2)&set(bybg));diff=[v2[b]-bybg[b] for b in common]
            paired[key][model]={'mean_marginal_minus_control':float(np.mean(diff)),
                'ci95_background':ci(diff),'n_backgrounds':len(common)}
    margins={}
    for r in csv.DictReader((out/'true_vs_wrong_class_margins.csv').open(encoding='utf-8')):
        margins.setdefault((r['regime'],str(r['N']),r['background_id']),[]).append(float(r['best_wrong_minus_true_margin']))
    margin_cells={}
    for (reg,n,bg),vals in margins.items():margin_cells.setdefault(f'{reg}|N={n}',[]).append(float(np.mean(vals)))
    margin_summary={k:{'mean':float(np.mean(v)),'ci95_background':ci(v),'n_backgrounds':len(v)} for k,v in margin_cells.items()}
    cond_bg={}
    for r in csv.DictReader((out/'true_vs_wrong_class_margins.csv').open(encoding='utf-8')):
        cond_bg.setdefault((r['regime'],str(r['N']),r['background_id']),[]).append(float(r['old_posterior_precision_condition_number']))
    condition_summary={}
    for (reg,n,bg),vals in cond_bg.items():condition_summary.setdefault(f'{reg}|N={n}',[]).append(float(np.mean(vals)))
    condition_summary={k:{'mean':float(np.mean(v)),'ci95_background':ci(v),'n_backgrounds':len(v)} for k,v in condition_summary.items()}
    per_prog_margins={}
    for r in csv.DictReader((out/'true_vs_wrong_class_margins.csv').open(encoding='utf-8')):
        per_prog_margins.setdefault((r['regime'],str(r['N']),r['background_id']),[]).append(float(r['best_wrong_minus_true_program_score']))
    byprog={}
    for (reg,n,bg),v in per_prog_margins.items():byprog.setdefault(f'{reg}|N={n}',[]).append(float(np.mean(v)))
    program_margin_summary={k:{'mean':float(np.mean(v)),'ci95_background':ci(v),'n_backgrounds':len(v)} for k,v in byprog.items()}
    bp=__import__('torch').load(out/'basis_pretrained.pt',map_location='cpu',weights_only=False)
    oracle_rows=list(csv.DictReader((out/'privileged_oracle_per_background.csv').open(encoding='utf-8')))
    oracle_bg={}
    for r in oracle_rows:oracle_bg.setdefault((r['regime'],str(r['N']),r['background_id'],r['oracle']),[]).append(float(r['functional_class_correct']))
    oracle_cells={}
    for (reg,n,bg,oracle),vals in oracle_bg.items():oracle_cells.setdefault((reg,n,oracle),{})[bg]=float(np.mean(vals))
    oracle_summary={}
    for (reg,n,oracle),bybg in oracle_cells.items():
        vals=list(bybg.values());oracle_summary.setdefault(f'{reg}|{oracle}|N={n}',{})['functional_class_accuracy']={
            'mean':float(np.mean(vals)),'ci95_background':ci(vals),'n_backgrounds':len(vals)}
    for n in cfg['budgets']:
      for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
        known=oracle_cells.get((reg,str(n),'known_old_privileged'),{});est=oracle_cells.get((reg,str(n),'estimated_old_privileged'),{})
        common=sorted(set(known)&set(est));diff=[known[k]-est[k] for k in common]
        if diff:oracle_summary[f'{reg}|known_minus_estimated|N={n}']={'mean':float(np.mean(diff)),'ci95_background':ci(diff),'n_backgrounds':len(diff)}
    fit=list(csv.DictReader((out/'privileged_old_fit.csv').open(encoding='utf-8')))
    fit_summary={}
    for reg in sorted({r['regime'] for r in fit}):
        part=[r for r in fit if r['regime']==reg]
        fit_summary[reg]={'mean_old_support_rmse':float(np.mean([float(r['old_support_rmse']) for r in part])),
            'mean_parameter_rmse':float(np.mean([float(r['parameter_rmse']) for r in part])),
            'mean_b_error':float(np.mean([float(r['b_error']) for r in part]))}
    # Independent query effect diagnostics from the main pipeline.
    primary=list(csv.DictReader((out/'per_background.csv').open(encoding='utf-8')))
    effect_summary={}
    for reg in sorted({r['regime'] for r in primary}):
      for pcode in ('[0]','[1]','[2]','[3]'):
        part=[r for r in primary if r['regime']==reg and r['true_program']==pcode and r['N']=='100']
        vals=[float(r['effect_skill']) for r in part if r['effect_skill'] not in ('','None')]
        if vals:effect_summary[f'{reg}|U{int(pcode[1])+1}']={'mean_effect_skill':float(np.mean(vals)),
            'n_background_seed_rows':len(vals),'mean_constant_drag_analytic_skill':(
                float(np.mean([float(r['analytical_u4_constant_drag_skill']) for r in part
                  if r['analytical_u4_constant_drag_skill'] not in ('','None')])) if pcode=='[3]' else None)}
    effect_controls={}
    for filename,protocol in [('results/q1/q1_v2_bayesian_operators_dev1/per_background.csv','LOW_EXCITATION_PRIMARY'),
                              ('results/q1/q1_v2_informative_dev1/per_code_effect.csv','INFORMATIVE_SECONDARY')]:
        rr=list(csv.DictReader((ROOT/filename).open(encoding='utf-8')))
        for reg in sorted({x['regime'] for x in rr}):
          for code in range(4):
            part=[x for x in rr if x['regime']==reg and
                  (x.get('true_program')=='['+str(code)+']' if 'true_program' in x else int(x['code'])==code) and
                  (x.get('N','100')=='100')]
            bvals={}
            for x in part:bvals.setdefault(x['background_id'],[]).append(float(x['effect_skill']))
            vals=[float(np.mean(v)) for v in bvals.values()]
            effect_controls[f'{protocol}|{reg}|U{code+1}']={'mean':float(np.mean(vals)),
                'ci95_background':ci(vals),'n_backgrounds':len(vals)}
    order_results={}
    for filename,protocol in [('results/q1/q1_v2_bayesian_operators_dev1/per_background.csv','LOW_EXCITATION_PRIMARY'),
                              ('results/q1/q1_v2_informative_dev1/per_background.csv','INFORMATIVE_SECONDARY')]:
        rr=list(csv.DictReader((ROOT/filename).open(encoding='utf-8')))
        for reg in sorted({x['regime'] for x in rr}):
          for pair in ('[0, 1]','[1, 0]'):
            part=[x for x in rr if x['regime']==reg and x['true_program']==pair and x['N']=='100']
            bvals={}
            for x in part:bvals.setdefault(x['background_id'],[]).append(float(x['functional_class_correct']))
            vals=[float(np.mean(v)) for v in bvals.values()]
            order_results[f'{protocol}|{reg}|{pair}']={'mean_class_accuracy':float(np.mean(vals)),
                'ci95_background':ci(vals),'n_backgrounds':len(vals)}
    # Numeric noncommutativity of the learned action block and their functional response.
    op_training=json.loads((ROOT/'results/q1/q1_v2_bayesian_operators_dev1/operator_training.json').read_text(encoding='utf-8'))
    cmap=op_training['opaque_code_map'];comm=[]
    for seed in cfg['optimizer_seeds']:
        bank=AffineOperatorBank();bank.load_state_dict(torch.load(ROOT/f'results/q1/q1_v2_bayesian_operators_dev1/operator_seed{seed}.pt',map_location='cpu',weights_only=False)['state_dict'])
        a0=bank.matrix(cmap['0']);a1=bank.matrix(cmap['1']);comm.append(float(torch.linalg.matrix_norm(a1@a0-a0@a1).detach()))
    inf_main=list(csv.DictReader((ROOT/'results/q1/q1_v2_informative_dev1/per_background.csv').open(encoding='utf-8')))
    inf_legacy=list(csv.DictReader((ROOT/'results/q1/q1_v2_informative_dev1/legacy_per_background.csv').open(encoding='utf-8')))
    inf_summary={}
    for reg in ('IDLIKE_DEVELOPMENT','BACKGROUND_OOD_DEVELOPMENT'):
      for n in cfg['budgets']:
        main_bg={}
        for x in inf_main:
            if x['regime']==reg and int(x['N'])==n:main_bg.setdefault(x['background_id'],[]).append(float(x['functional_class_correct']))
        legacy_bg={}
        for x in inf_legacy:
            if x['regime']==reg and x['model']=='M4' and int(x['N'])==n:legacy_bg.setdefault(x['background_id'],[]).append(float(x['functional_class_correct']))
        m4_bg={f'{reg}-{bg.rsplit("-",1)[1]}':v for bg,v in legacy_bg.items()}
        ids=sorted(set(main_bg)&set(m4_bg));ma=[float(np.mean(main_bg[i])) for i in ids];le=[float(np.mean(m4_bg[i])) for i in ids];diff=[x-y for x,y in zip(ma,le)]
        inf_summary[f'{reg}|N={n}']={'v2_accuracy':{'mean':float(np.mean(ma)),'ci95_background':ci(ma)},
          'legacy_M4_accuracy':{'mean':float(np.mean(le)),'ci95_background':ci(le)},
          'paired_v2_minus_M4':{'mean':float(np.mean(diff)),'ci95_background':ci(diff),'n_backgrounds':len(ids)}}
    obj={'confidence_unit':'background; programs and optimizer seeds averaged within background',
      'functional_class_controls':cells,'paired_differences':paired,
      'true_vs_best_wrong_class_score_margin':margin_summary,
      'best_wrong_class_minus_true_program_score':program_margin_summary,
      'old_posterior_precision_condition_number':condition_summary,
      'privileged_oracles':oracle_summary,'privileged_old_fit':fit_summary,
      'independent_query_effect_skill':effect_summary,'effect_skill_background_intervals':effect_controls,
      'noncommuting_order_pair_accuracy':order_results,'learned_action_block_commutator_norm_by_seed':comm,
      'informative_secondary_paired_legacy_M4':inf_summary,
      'prior_covariance_trace':float(__import__('torch').trace(bp['prior_covariance'])),
      'score_margin_convention':'best wrong-class NLL minus best true-class NLL; positive favors the true class'}
    (out/'metrics_extended.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    metrics_path=out/'metrics.json';metrics_obj=json.loads(metrics_path.read_text(encoding='utf-8'))
    for key,cell in metrics_obj['metrics'].items():
        cell.pop('mean_true_vs_best_wrong_class_margin',None)
        margin=margin_summary.get(key)
        if margin:cell['mean_best_wrong_class_minus_best_true_class_nll']=margin['mean']
    metrics_obj['limitations']=[
      'Development diagnostic only; original test splits were not used.',
      'Assessment query labels are used only after program selection for query-metric reporting.',
      'Legacy M2/M3/M4 and privileged known-old/estimated-old oracles are in separate paired artifacts.',
      'The informative-exploration protocol is a separate secondary development study; not confirmatory.']
    metrics_path.write_text(json.dumps(metrics_obj,indent=2),encoding='utf-8')
    per_bg_path=out/'per_background.csv'
    persisted=list(csv.DictReader(per_bg_path.open(encoding='utf-8')))
    if persisted and 'true_vs_best_wrong_class_margin' in persisted[0]:
        for row in persisted:
            row['best_wrong_class_minus_top_ranked_score']=row.pop('true_vs_best_wrong_class_margin')
        with per_bg_path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(persisted[0]));writer.writeheader();writer.writerows(persisted)
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        return h.hexdigest()
    primary_sources=['q1_v2/basis.py','q1_v2/bayes.py','q1_v2/operators.py','q1_v2/inference.py','q1_v2/training.py','scripts/q1_v2_pipeline.py','scripts/q1_v2_legacy_compare.py','scripts/q1_v2_privileged_oracles.py','scripts/q1_v2_finalize_report.py','scripts/q1_v2_summarize.py']
    (out/'provenance.posthoc.json').write_text(json.dumps({'pipeline_provenance_file':'provenance.json',
      'analysis_sources_sha256':{x:sha(ROOT/x) for x in primary_sources},
      'legacy_checkpoint_sha256':{x:sha(ROOT/x) for x in ['results/q1/cpu_confirmatory_impl/m2/context_model.pt','results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt','results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt']},
      'posthoc_artifact_sha256':{x:sha(out/x) for x in ['per_background.csv','legacy_per_background.csv','legacy_comparison.json','privileged_oracle_per_background.csv','privileged_old_fit.csv','true_vs_wrong_class_margins.csv','metrics_extended.json']},
      'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu',
      'test_split_used':False},indent=2),encoding='utf-8')
    informative=ROOT/'results/q1/q1_v2_informative_dev1'
    inf_metrics=json.loads((informative/'metrics.json').read_text(encoding='utf-8'))
    inf_metrics['limitations']=['Secondary development only; original test splits were not used.',
      'Paired legacy M3/M4 comparison is in legacy_metrics.json.',
      'Informative secondary point-estimate, no-composition, context-ablated ranking controls and privileged system-ID references were not run.']
    (informative/'metrics.json').write_text(json.dumps(inf_metrics,indent=2),encoding='utf-8')
    (informative/'provenance.json').write_text(json.dumps({'protocol_config_sha256':sha(ROOT/'configs/q1_v2_informative.yaml'),
      'primary_split_manifest_sha256':sha(out/'split_manifest.json'),
      'selected_schedule_sha256':sha(informative/'schedule_selection.json'),
      'informative_scripts_sha256':{x:sha(ROOT/x) for x in ['scripts/q1_v2_informative.py','scripts/q1_v2_informative_legacy.py','scripts/q1_v2_effect_controls.py']},
      'v2_checkpoint_sha256':{x:sha(out/x) for x in ['basis_pretrained.pt','operator_seed4201.pt','operator_seed4202.pt','context_ablated_operator_seed4201.pt','context_ablated_operator_seed4202.pt']},
      'legacy_checkpoint_sha256':{x:sha(ROOT/x) for x in ['results/q1/cpu_confirmatory_impl/m2/context_model.pt','results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt','results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt']},
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'device':'cpu'},
      'test_split_used':False},indent=2),encoding='utf-8')
    print(f"Summarized {len(records)} same-case control rows and {sum(map(len,margins.values()))} margin rows")
if __name__=='__main__':main()
