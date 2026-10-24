from __future__ import annotations
import ast
from pathlib import Path

import torch

from q1_v2.basis import PartitionedBasis
from q1_v2.inference import composed
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import (ObservedEffectTask, ObservedEpisode, design,
    evaluate_effect_tasks, fit_operator_bank, fit_operator_bank_effect_aware)


def _episodes(k=12):
    gen=torch.Generator().manual_seed(111)
    out=[]
    for code in range(4):
        sx=torch.randn(8,6,generator=gen);qx=torch.randn(8,6,generator=gen)
        out.append(ObservedEpisode(sx,torch.randn(8,2,generator=gen),qx,
            torch.randn(8,2,generator=gen),(code,)))
    return out


def _effect_tasks(seed=333):
    gen=torch.Generator().manual_seed(seed);tasks=[]
    for code in range(4):
        for bg in range(3):
            w=torch.randn(12,generator=gen)*.1
            delta=torch.zeros(12)
            if code==0:delta[2]=.012
            if code==1:delta[3]=-.018
            if code==2:delta[0]=.035
            if code==3:delta[1]=-.012
            for fold in range(2):
                tasks.append(ObservedEffectTask(code,w,delta,torch.randn(24,6,generator=gen),f'bg{bg}',fold))
    return tasks


def _matched_synthetic_data(basis,seed=555,backgrounds=3):
    gen=torch.Generator().manual_seed(seed);episodes=[];tasks=[]
    for bg in range(backgrounds):
        w=torch.randn(12,generator=gen)*.08
        for code in range(4):
            delta=torch.zeros(12);delta[code]=[.008,-.009,.015,-.006][code]
            sx=torch.randn(16,6,generator=gen);qx=torch.randn(16,6,generator=gen)
            with torch.no_grad():
                sy=(basis(sx)@w).detach();qy=(basis(qx)@(w+delta)).detach()
            episodes.append(ObservedEpisode(sx,sy,qx,qy,(code,)))
            for fold in range(2):
                px=torch.randn(32,6,generator=gen)
                tasks.append(ObservedEffectTask(code,w.clone(),delta.clone(),px,f'bg{bg}',fold))
    return episodes,tasks


def test_partitioned_basis_width_candidates_keep_additive_blocks_and_gradients():
    x=torch.randn(9,6,requires_grad=True)
    for widths,k in [((6,4,2),12),((6,8,2),16)]:
        basis=PartitionedBasis(widths=widths,hidden=8)
        y=basis(x)
        assert y.shape==(9,2,k)
        y.square().mean().backward(retain_graph=True)
        assert torch.isfinite(x.grad).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in basis.parameters())
        x.grad.zero_()


def test_design_keeps_delta_v_coordinate_order_and_units():
    class CoordinateBasis(torch.nn.Module):
        def forward(self,x):
            value=torch.zeros(*x.shape[:-1],2,2,dtype=x.dtype)
            value[...,0,0]=x[...,0];value[...,0,1]=1
            value[...,1,0]=2*x[...,1];value[...,1,1]=3
            return value
    x=torch.tensor([[2.,5.,0,0,0,0],[7.,11.,0,0,0,0]])
    y=torch.tensor([[13.,17.],[19.,23.]])
    phi=design(CoordinateBasis(),x)
    assert torch.equal(phi,torch.tensor([[2.,1.],[10.,3.],[7.,1.],[22.,3.]]))
    assert torch.equal(y.reshape(-1),torch.tensor([13.,17.,19.,23.]))


def test_opaque_mapping_and_affine_composition_preserve_order_and_bias():
    bank=AffineOperatorBank(widths=(1,1,1),codes=4)
    with torch.no_grad():
        for block in bank.blocks:
            block[0,0,0]=2.;block[1,0,0]=3.
        bank.bias[0]=torch.tensor([1.,0.,0.]);bank.bias[1]=torch.tensor([0.,2.,0.])
    physical_to_opaque={0:1,1:0}
    result=composed(bank,(physical_to_opaque[0],physical_to_opaque[1]))
    first=(bank.matrix(physical_to_opaque[0]),bank.bias[physical_to_opaque[0]])
    second=(bank.matrix(physical_to_opaque[1]),bank.bias[physical_to_opaque[1]])
    expected=AffineOperatorBank.compose(first,second)
    assert torch.allclose(result[0],expected[0])
    assert torch.allclose(result[1],expected[1])
    reverse=composed(bank,(physical_to_opaque[1],physical_to_opaque[0]))
    assert not torch.allclose(result[1],reverse[1])


def test_dynamic_basis_operator_dimension_and_per_observation_objective():
    basis=PartitionedBasis(widths=(6,8,2),hidden=8)
    episodes=_episodes()
    inner=_episodes()
    bank,stats=fit_operator_bank(basis,torch.zeros(16),torch.eye(16),episodes,
        sigma2=.01,seed=87,epochs=1,inner_episodes=inner,normalize_nll_per_observation=True)
    assert tuple(bank.widths)==(6,8,2)
    assert bank.matrix(0).shape==(16,16)
    assert stats['nll_normalized_per_observation'] is True
    assert stats['epoch_coverage'][0]['training_episode_count']==len(episodes)


def test_effect_aware_training_learns_small_in_basis_effects_and_selects_on_inner_only():
    basis=PartitionedBasis(hidden=8)
    train_eps,train_tasks=_matched_synthetic_data(basis,333)
    inner_eps,inner_tasks=_matched_synthetic_data(basis,444)
    scales={c:1e-4 for c in range(4)}
    init=AffineOperatorBank()
    before=evaluate_effect_tasks(basis,init,inner_tasks,scales)
    bank,stats=fit_operator_bank_effect_aware(basis,torch.zeros(12),torch.eye(12),train_eps,
        train_tasks,effect_scales=scales,sigma2=.01,seed=902,epochs=6,lr=.001,
        regularization=1e-5,effect_weight=1.,inner_episodes=inner_eps,
        inner_effect_tasks=inner_tasks,inner_effect_scales=scales,
        selection_code_groups={'U1':0,'U2':1,'U3':2,'U4':3})
    after=evaluate_effect_tasks(basis,bank,inner_tasks,scales)
    assert after[2]['skill']>before[2]['skill']
    assert after[3]['skill']>before[3]['skill']
    assert stats['selection_split']=='inner-training backgrounds only'
    assert stats['best_epoch']==max((r for r in stats['curve'] if r['inner_U12_guard_pass']),
                                    key=lambda r:r['combined_selection_score'])['epoch']
    assert all(row['nll_episode_count']==len(train_eps) and row['unique_nll_episodes']==len(train_eps)
               for row in stats['epoch_coverage'])
    assert all(set(row['effect_task_counts_by_code'].values())=={6} for row in stats['epoch_coverage'])
    assert stats['inner_effect_scales_train_inner']=={str(k):v for k,v in scales.items()}


def test_seeded_effect_aware_runs_are_reproducible_and_independent():
    basis=PartitionedBasis(hidden=8);eps,tasks=_matched_synthetic_data(basis,77);inner_eps,inner=_matched_synthetic_data(basis,99);scales={c:1e-4 for c in range(4)}
    def fit(seed):
        return fit_operator_bank_effect_aware(basis,torch.zeros(12),torch.eye(12),eps,tasks,
            effect_scales=scales,sigma2=.01,seed=seed,epochs=2,lr=.001,regularization=1e-5,
            effect_weight=1.,inner_episodes=inner_eps,inner_effect_tasks=inner,inner_effect_scales=scales,
            selection_code_groups={'U1':0,'U2':1,'U3':2,'U4':3})
    a,sa=fit(4201);again,sagain=fit(4201);b,sb=fit(4202)
    assert all(torch.equal(a.state_dict()[k],again.state_dict()[k]) for k in a.state_dict())
    assert sa['curve']==sagain['curve']
    assert any(not torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    assert sa['curve']!=sb['curve']


def test_training_modules_keep_strict_simulator_boundary():
    for path in (Path(__file__).parents[1]/'q1_v2').glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        imported={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        imported|={alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
        assert 'q1.simulator' not in imported
        assert not any(name in {'response','apply_program','intervention','DynamicsParams'} for name in imported)
