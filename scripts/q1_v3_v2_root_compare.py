"""Score frozen V2 root-repair banks on the exact fresh V3 development cases."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from q1.datasets import stage_pair
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS,DynamicsParams,response,apply_program
from q1_v2.basis import PartitionedBasis
from q1_v2.bayes import gaussian_posterior
from q1_v2.inference import composed,rank_programs
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import design
from scripts.q1_v2_pipeline import rows

def main():
    run=ROOT/'results/q1/q1_v3_mechanism_dev1';v2=ROOT/'results/q1/q1_v2_root_repair_dev3'
    cfg=yaml.safe_load((run/'config.resolved.yaml').read_text(encoding='utf-8'))
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'));split=protocol['split_manifest']
    selected=float(json.loads((run/'length_penalty_selection.json').read_text(encoding='utf-8'))['selected'])
    device=torch.device('cpu');torch.set_num_threads(1)
    bd=torch.load(v2/'basis_pretrained.pt',map_location=device,weights_only=False)
    basis=PartitionedBasis(hidden=32).to(device);basis.load_state_dict(bd['basis']);basis.eval()
    for p in basis.parameters():p.requires_grad_(False)
    banks={};maps={}
    for seed in cfg['optimizer_seeds']:
        saved=torch.load(v2/f'operator_seed{seed}_effect_aware.pt',map_location=device,weights_only=False)
        bank=AffineOperatorBank().to(device);bank.load_state_dict(saved['state_dict']);bank.eval();banks[int(seed)]=bank;maps[int(seed)]={int(k):int(v) for k,v in saved['opaque_code_map'].items()}
    out=[]
    for regime,key in [('IDLIKE_DEVELOPMENT','development_idlike'),('BACKGROUND_OOD_DEVELOPMENT','development_background_ood')]:
      for bi,(bgrow,seed) in enumerate(zip(split[key]['backgrounds'],split[key]['seeds'])):
        p=DynamicsParams(np.asarray(bgrow['B']),float(bgrow['d']),float(bgrow['b']),float(bgrow['kappa']));classes=functional_classes(p,dt=cfg['dt'])[0]
        for pi,program in enumerate(PROGRAMS):
          old,adapt,query=stage_pair(p,program,seed=int(seed)+300000+pi*7919,n_support=96,n_query=cfg['adaptation_trajectory'],dt=cfg['dt'],sigma=cfg['sigma'])
          ox,oy=rows(old);ax,ay=rows(adapt);qx,qy=rows(query)
          ox=torch.as_tensor(ox,dtype=torch.float32);oy=torch.as_tensor(oy,dtype=torch.float32);ax=torch.as_tensor(ax,dtype=torch.float32);ay=torch.as_tensor(ay,dtype=torch.float32)
          qx=torch.as_tensor(qx,dtype=torch.float32);qy=torch.as_tensor(qy,dtype=torch.float32)
          for seed0,bank in banks.items():
            cmap=maps[seed0];candidates=[tuple(cmap[c] for c in prog) for prog in PROGRAMS]
            prior=gaussian_posterior(design(basis,ox),oy,bd['prior_mean'],bd['prior_covariance'],float(bd['sigma2']))
            qphi=design(basis,qx)
            true_effect=None
            if len(program)==1:true_effect=response(apply_program(p,program),qx.numpy(),dt=cfg['dt'])-response(p,qx.numpy(),dt=cfg['dt'])
            for n in cfg['budgets']:
              ranked=rank_programs(bank,prior,candidates,design(basis,ax[:n]),ay[:n],float(bd['sigma2']),selected)
              point=rank_programs(bank,prior,candidates,design(basis,ax[:n]),ay[:n],float(bd['sigma2']),selected,point_estimate=True)
              for scoring,result in [('marginal',ranked),('point',point)]:
                opaque=result[0][0];pred=tuple(next(k for k,v in cmap.items() if v==code) for code in opaque)
                am,b=composed(bank,opaque)
                with torch.no_grad():predv=(qphi@(am@prior.mean+b)).reshape(-1,2)
                mse=float((predv-qy).square().mean().detach())
                skill=None
                if true_effect is not None:
                  code=cmap[program[0]];a1,b1=composed(bank,(code,));eff=(qphi@((a1@prior.mean+b1)-prior.mean)).reshape(-1,2).detach().numpy()
                  energy=float(np.mean(true_effect**2));skill=1-float(np.mean((eff-true_effect)**2))/max(energy,1e-30)
                out.append({'regime':regime,'background_id':f'{regime}-{bi:03d}','background_seed':int(seed),'true_program':json.dumps(program),'N':n,
                  'optimizer_seed':seed0,'model':'V2_ROOT_REPAIR','scoring':scoring,'predicted_program':json.dumps(pred),
                  'functional_class_correct':int(classes[PROGRAMS.index(pred)]==classes[pi]),'raw_program_correct':int(pred==program),
                  'query_delta_v_mse':mse,'effect_skill':skill,'map_theta_trace_covariance':float(torch.trace(prior.covariance))})
      print(f'Scored root V2 {regime} {bi+1}/16',flush=True)
    with (run/'v2_root_repair_per_background.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print(f'Wrote {len(out)} paired V2 root-repair rows')
if __name__=='__main__':main()
