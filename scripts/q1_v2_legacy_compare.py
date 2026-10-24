"""Paired v1 M2/M3/M4 evaluation on exactly the v2 development cases."""
from __future__ import annotations
import csv, json, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np, torch, yaml
from q1.datasets import stage_pair
from q1.evaluate import _public_transitions, _support_tensor
from q1.inference import rank_programs
from q1.model import ContextDynamicsModel
from q1.oracle import functional_classes
from q1.primitives import PrimitiveBank
from q1.simulator import PROGRAMS, DynamicsParams
from q1.train_primitives import _get_m2

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default='results/q1/q1_v2_bayesian_operators_dev1');a=p.parse_args()
    out=ROOT/a.run_dir
    cfg=yaml.safe_load((out/'config.resolved.yaml').read_text(encoding='utf-8'))
    manifest=json.loads((out/'split_manifest.json').read_text(encoding='utf-8'))
    devrows=[]
    m2path=ROOT/'results/q1/cpu_confirmatory_impl/m2/context_model.pt'
    m3path=ROOT/'results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt'
    m4path=ROOT/'results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt'
    device=torch.device('cpu'); torch.set_num_threads(1)
    m2,normalizer,m2data=_get_m2(m2path,device)
    m3data=torch.load(m3path,map_location=device,weights_only=False)
    m4data=torch.load(m4path,map_location=device,weights_only=False)
    code_map=m4data['group_to_code']
    context_dim=int(m2data['config']['model']['context_dim'])
    m3bank=PrimitiveBank(context_dim,8,64).to(device);m3bank.load_state_dict(m3data['primitives']);m3bank.eval()
    m4bank=PrimitiveBank(context_dim,8,64).to(device);m4bank.load_state_dict(m4data['primitives']);m4bank.eval()
    inverse={int(v):int(k) for k,v in code_map.items()}
    for regime in ('development_idlike','development_background_ood'):
        for bi,bg in enumerate(manifest[regime]['backgrounds']):
            before=DynamicsParams(np.asarray(bg['B']),float(bg['d']),float(bg['b']),float(bg['kappa']))
            seed=int(manifest[regime]['seeds'][bi]);classes,_=functional_classes(before,dt=cfg['dt'])
            for pi,program in enumerate(PROGRAMS):
                old,adapt,query=stage_pair(before,program,seed=seed+300000+pi*7919,
                    n_support=cfg['old_support'],n_query=max(cfg['adaptation_trajectory'],cfg['query_trajectory']),
                    dt=cfg['dt'],sigma=cfg['sigma'])
                context=m2.encode(_support_tensor(old.states,old.actions,normalizer,device))
                adapt_rows=_public_transitions(adapt.states,adapt.actions)
                query_rows=_public_transitions(query.states,query.actions)
                x=torch.as_tensor(query_rows[:,:6],dtype=torch.float32)
                y=torch.as_tensor(query_rows[:,6:],dtype=torch.float32)
                with torch.no_grad():
                    oldpred=normalizer.denormalize_dv(m2.executor(normalizer.normalize_x(x),context.expand(len(x),-1)))
                    old_mse=float(torch.mean((oldpred-y)**2))
                for n in cfg['budgets']:
                    vals={}
                    for name,bank,penalty in [('M3',m3bank,float(m4data['length_penalty'])),
                                               ('M4',m4bank,float(m4data['length_penalty']))]:
                        ranked,_=rank_programs(context,torch.as_tensor(adapt_rows[:n],dtype=torch.float32),
                            encoder=m2.encoder,executor=m2.executor,primitives=bank,normalizer=normalizer,
                            length_penalty=penalty,device=device)
                        opaque=ranked[0];predp=tuple(inverse[int(code)] for code in opaque)
                        with torch.no_grad():
                            qpred=normalizer.denormalize_dv(m2.executor(normalizer.normalize_x(x),
                                                                        bank.apply_program(context,opaque)))
                            mse=float(torch.mean((qpred-y)**2))
                        vals[name]=(predp,mse)
                    for name,(predp,mse) in vals.items():
                        devrows.append({'regime':regime,'background_id':f'{regime}-{bi:03d}',
                            'background_seed':seed,'true_program':json.dumps(program),'N':n,'model':name,
                            'predicted_program':json.dumps(predp),
                            'functional_class_correct':int(classes[PROGRAMS.index(predp)]==classes[pi]),
                            'query_delta_v_mse':mse,'m2_old_context_query_mse':old_mse})
    with (out/'legacy_per_background.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(devrows[0]));w.writeheader();w.writerows(devrows)
    summary={}
    for r in devrows:
        k=f"{r['regime']}|{r['model']}|N={r['N']}";summary.setdefault(k,[]).append(r)
    aggregate={k:{'n_background_program_cases':len(v),
        'functional_class_accuracy':float(np.mean([x['functional_class_correct'] for x in v])),
        'mean_query_delta_v_mse':float(np.mean([x['query_delta_v_mse'] for x in v])),
        'mean_m2_old_context_query_mse':float(np.mean([x['m2_old_context_query_mse'] for x in v]))}
        for k,v in summary.items()}
    report={'paired_same_cases':True,'same_old_support_adaptation_and_assessment_query_seeds':True,
        'checkpoints':{str(x.relative_to(ROOT)):sha(x) for x in (m2path,m3path,m4path)},
        'metrics':aggregate,'m2_role':'empty-program old-context prediction baseline; M2 is not a program composition model',
        'direct_G':'not included: v1 direct-effect checkpoints were trained with privileged paired simulator effects, and do not supply a matched weak-label v2 intervention learner'}
    (out/'legacy_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Wrote paired legacy comparison ({len(devrows)} rows) to {out}')
if __name__=='__main__':main()
