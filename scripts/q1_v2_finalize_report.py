"""Efficiently finalize background-clustered v2 controls and score margins."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from scipy.stats import t as student_t
from q1.datasets import stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS,DynamicsParams
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import gaussian_posterior
from q1_v2.inference import rank_programs
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import design
from scripts.q1_v2_pipeline import rows,tensor_input

def ci(xs):
    x=np.asarray(xs,float);n=len(x)
    if n<2:return [float(x.mean()),float(x.mean())]
    return [float(x.mean()-student_t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)),float(x.mean()+student_t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n))]
def prog(s):return tuple(json.loads(s))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='results/q1/q1_v2_bayesian_operators_dev1');a=p.parse_args();out=ROOT/a.run_dir
    cfg=yaml.safe_load((out/'config.resolved.yaml').read_text(encoding='utf-8'));manifest=json.loads((out/'split_manifest.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1);bp=torch.load(out/'basis_pretrained.pt',map_location='cpu',weights_only=False)
    basis=PartitionedBasis(hidden=cfg['hidden']);basis.load_state_dict(bp['basis']);basis.eval()
    for z in basis.parameters():z.requires_grad_(False)
    mean,cov,sigma2=bp['prior_mean'],bp['prior_covariance'],float(bp['sigma2'])
    cmap=json.loads((out/'operator_training.json').read_text(encoding='utf-8'))['opaque_code_map'];inv={int(v):int(k) for k,v in cmap.items()}
    banks={};
    for seed in cfg['optimizer_seeds']:
        b=AffineOperatorBank();b.load_state_dict(torch.load(out/f'operator_seed{seed}.pt',map_location='cpu',weights_only=False)['state_dict']);b.eval()
        for z in b.parameters():z.requires_grad_(False)
        banks[seed]=b
    primary=list(csv.DictReader((out/'per_background.csv').open(encoding='utf-8')))
    legacy=list(csv.DictReader((out/'legacy_per_background.csv').open(encoding='utf-8')))
    class_map={}; control_rows=[]
    for reg in ('development_idlike','development_background_ood'):
        name='IDLIKE_DEVELOPMENT' if reg=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
        for bi,bg in enumerate(manifest[reg]['backgrounds']):
            params=DynamicsParams(np.asarray(bg['B']),float(bg['d']),float(bg['b']),float(bg['kappa']))
            class_map[f'{name}-{bi:03d}']=functional_classes(params,dt=cfg['dt'])[0]
    for r in primary:
        classes=class_map[r['background_id']];truep=prog(r['true_program']);trueclass=classes[PROGRAMS.index(truep)]
        for col,label in [('predicted_program','marginal_correct'),('point_predicted_program','point_correct'),
            ('single_only_predicted_program','single_only_correct'),('wrong_code_predicted_program','wrong_code_correct'),
            ('context_ablated_predicted_program','context_ablated_correct')]:
            pred=prog(r[col]);control_rows.append({**r,'model':label,
                'correct':int(classes[PROGRAMS.index(pred)]==trueclass)})
        control_rows.append({**r,'model':'identity_only_correct',
            'correct':int(classes[0]==trueclass)})
    for r in legacy:
        regime='IDLIKE_DEVELOPMENT' if r['regime']=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
        idx=int(r['background_id'].rsplit('-',1)[1]);bgid=f'{regime}-{idx:03d}'
        control_rows.append({**r,'regime':regime,'background_id':bgid,
            'model':r['model']+'_correct','correct':int(r['functional_class_correct'])})
    # Recompute only the true-class vs best-wrong-class score margin; controls above are
    # derived from the already persisted, same-seed candidate selections.
    margins=[];started=time.time()
    for reg in ('development_idlike','development_background_ood'):
        name='IDLIKE_DEVELOPMENT' if reg=='development_idlike' else 'BACKGROUND_OOD_DEVELOPMENT'
        for bi,bg in enumerate(manifest[reg]['backgrounds']):
            background_id=f'{name}-{bi:03d}';params=DynamicsParams(np.asarray(bg['B']),float(bg['d']),float(bg['b']),float(bg['kappa']))
            classes=class_map[background_id];seed0=int(manifest[reg]['seeds'][bi])
            for pi,truep in enumerate(PROGRAMS):
                old,adapt,_=stage_pair(params,truep,seed=seed0+300000+pi*7919,
                    n_support=cfg['old_support'],n_query=cfg['adaptation_trajectory'],dt=cfg['dt'],sigma=cfg['sigma'])
                ox,oy=rows(old);ax,ay=rows(adapt);post=gaussian_posterior(design(basis,tensor_input(ox)),tensor_input(oy),mean,cov,sigma2)
                phi=design(basis,tensor_input(ax));target=tensor_input(ay).reshape(-1);truecl=classes[pi]
                code_programs=[tuple(cmap[str(c)] for c in pr) for pr in PROGRAMS]
                for optseed,bank in banks.items():
                    for n in cfg['budgets']:
                        ranked=rank_programs(bank,post,code_programs,phi[:2*n],target[:2*n],sigma2,0.0)
                        score_by_class={}
                        for cp,score,_ in ranked:
                            pp=tuple(inv[c] for c in cp);cl=classes[PROGRAMS.index(pp)]
                            score_by_class[cl]=min(score_by_class.get(cl,float('inf')),float(score.detach()))
                        true_score=score_by_class[truecl]
                        wrong_score=min(v for cl,v in score_by_class.items() if cl!=truecl)
                        true_opaque=tuple(cmap[str(c)] for c in truep)
                        true_program_score=next(float(score.detach()) for cp,score,_ in ranked if cp==true_opaque)
                        margins.append({'regime':name,'background_id':background_id,'optimizer_seed':optseed,
                          'true_program':json.dumps(truep),'N':n,'true_class_score':true_score,
                          'best_wrong_class_score':wrong_score,'best_wrong_minus_true_margin':wrong_score-true_score,
                          'true_program_score':true_program_score,
                          'best_wrong_minus_true_program_score':wrong_score-true_program_score,
                          'old_posterior_precision_condition_number':post.precision_condition_number,
                          'old_posterior_cholesky_jitter':post.jitter})
    with (out/'true_vs_wrong_class_margins.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(margins[0]));w.writeheader();w.writerows(margins)
    # Means and paired differences use background as the independent unit.
    bg_values={}
    for r in control_rows:
        key=(r['regime'],str(r['N']),r['background_id'],r['model'])
        bg_values.setdefault(key,[]).append(float(r['correct']))
    bg_summary={}
    for (reg,n,bg,model),vals in bg_values.items():bg_summary.setdefault((reg,n,model),{})[bg]=float(np.mean(vals))
    cells={}
    for (reg,n,model),bybg in bg_summary.items():
        vals=list(bybg.values());cells.setdefault(f'{reg}|N={n}',{})[model]={'mean':float(np.mean(vals)),
            'ci95_background':ci(vals),'n_backgrounds':len(vals)}
    paired={}
    for key,methods in cells.items():
        v2=bg_summary[(key.split('|N=')[0],key.split('|N=')[1],'marginal_correct')]
        paired[key]={}
        for m,bybg in bg_summary.items():
            if m[0]+'|N='+m[1]!=key or m[2]=='marginal_correct':continue
            ids=sorted(set(v2)&set(bybg));d=[v2[i]-bybg[i] for i in ids]
            paired[key][m[2]]={'mean_marginal_minus_control':float(np.mean(d)),'ci95_background':ci(d),'n_backgrounds':len(d)}
    margin_bg={}
    for r in margins:margin_bg.setdefault((r['regime'],str(r['N']),r['background_id']),[]).append(r['best_wrong_minus_true_margin'])
    margin_summary={}
    for (reg,n,bg),vals in margin_bg.items():margin_summary.setdefault(f'{reg}|N={n}',[]).append(float(np.mean(vals)))
    margin_summary={k:{'mean':float(np.mean(v)),'ci95_background':ci(v),'n_backgrounds':len(v)} for k,v in margin_summary.items()}
    obj={'confidence_unit':'background; programs and optimizer seeds averaged within background',
      'functional_class_controls':cells,'paired_differences':paired,'true_vs_best_wrong_class_score_margin':margin_summary,
      'prior_covariance_trace':float(torch.trace(cov)),
      'runtime_margin_recompute_seconds':time.time()-started,
      'score_margin_convention':'best wrong-class NLL minus best true-class NLL; positive favors the true class'}
    (out/'metrics_extended.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(f"Finalized paired controls and margins ({len(margins)} scored rows) in {time.time()-started:.1f}s")
if __name__=='__main__':main()
