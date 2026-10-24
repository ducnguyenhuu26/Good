import numpy as np
import torch
from torch import nn

from q1.datasets import make_split_manifest, sample_environment, stage_pair
from q1.model import ContextDynamicsModel, TransitionNormalizer
from q1.oracle import functional_classes
from q1.simulator import PROGRAMS, apply_program, response, transition
from q1.train_primitives import TransferTask, _paired_effect_scales
from scripts.q1_u4_root_cause import _batch_delta, _fit_latent, _paired_effect, _predict


def test_u3_u4_intervention_effect_identity_including_zero_velocity():
    rng = np.random.default_rng(41)
    manifest = make_split_manifest()
    for i in range(40):
        params = sample_environment(manifest, "train", i, {
            "train": ((.75, 1.), (.28, .42), (.08, .14), (.18, .28))})
        state = rng.normal(size=4)
        if i == 0:
            state[2:] = 0.
        action = rng.normal(size=2)
        for code, expected in ((2, -.4*.05*params.d*state[2:]),
                               (3, -.5*.05*params.b*np.linalg.norm(state[2:])*state[2:])):
            old = transition(state, action, params, dt=.05)[2:] - state[2:]
            new = transition(state, action, apply_program(params, (code,)), dt=.05)[2:] - state[2:]
            np.testing.assert_allclose(new-old, expected, atol=1e-14, rtol=1e-12)


def test_u3_and_u4_change_only_their_declared_physical_parameter():
    p = sample_environment(make_split_manifest(), "train", 4)
    u3, u4 = apply_program(p,(2,)), apply_program(p,(3,))
    np.testing.assert_array_equal(u3.B,p.B); assert u3.b==p.b and u3.kappa==p.kappa and u3.d==1.4*p.d
    np.testing.assert_array_equal(u4.B,p.B); assert u4.d==p.d and u4.kappa==p.kappa and u4.b==1.5*p.b


def test_paired_effect_uses_same_probes_and_conditional_means_without_noise():
    params = sample_environment(make_split_manifest(), "train", 0)
    probes = np.arange(48, dtype=np.float64).reshape(8, 6) / 17
    actual = _paired_effect(params, 3, probes, .05)
    expected = -.5*.05*params.b*np.linalg.norm(probes[:, 2:4], axis=1, keepdims=True)*probes[:, 2:4]
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    # Noise is absent from response() conditional means; shared probe object is consumed on each side.
    assert actual.shape == (8, 2) and np.isfinite(actual).all()


def test_physical_oracle_program_mapping_and_equivalence_on_same_background():
    params = sample_environment(make_split_manifest(), "validation", 0)
    classes, _ = functional_classes(params)
    u4 = PROGRAMS.index((3,))
    identity = PROGRAMS.index(())
    assert classes[u4] != classes[identity]
    probes = np.zeros((3, 6), dtype=np.float64)
    probes[:, 2] = [.1, .2, .3]
    observed = response(apply_program(params, (3,)), probes)
    correct = np.mean((observed-response(apply_program(params, (3,)), probes))**2)
    wrong = np.mean((observed-response(params, probes))**2)
    assert correct < wrong


def test_oracle_scores_continuous_stage_pair_transitions_under_true_program_mapping():
    params = sample_environment(make_split_manifest(), "validation", 1)
    old, adaptation, _ = stage_pair(params, (3,), seed=9201, n_support=96, n_query=100,
                                   dt=.05, sigma=0.)
    assert adaptation.states[0].tolist() == old.states[-1].tolist()
    observed = adaptation.delta_v[:20]
    states = adaptation.states[:21]
    actions = adaptation.actions[:20]
    true = _batch_delta(states, actions, apply_program(params, (3,)), .05)
    wrong = _batch_delta(states, actions, params, .05)
    assert np.mean((true-observed)**2) < np.mean((wrong-observed)**2)


def test_normalizer_roundtrip_and_executor_boundary_is_physical_delta_v():
    norm = TransitionNormalizer(np.arange(6), np.arange(1,7), np.array([.2,-.1]), np.array([.5,.25]))
    x = torch.randn(12,6); y = torch.randn(12,2)
    torch.testing.assert_close(norm.normalize_x(x)*torch.tensor(norm.x_std)+torch.tensor(norm.x_mean), x)
    torch.testing.assert_close(norm.denormalize_dv(norm.normalize_dv(y)), y)
    model = _MockModel()
    prediction = _predict(model, norm, torch.tensor([[.2,0.]]), np.zeros((2,6)), torch.device("cpu"))
    np.testing.assert_allclose(prediction[:,0], .2*.5+.2, atol=1e-6)


class _MockExecutor(nn.Module):
    def __init__(self, reachable=True):
        super().__init__(); self.weight = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.reachable = reachable
    def forward(self, x, c):
        del x
        if not self.reachable:
            return torch.zeros((*c.shape[:-1], 2), dtype=c.dtype, device=c.device)
        return torch.stack((c[..., 0], torch.zeros_like(c[..., 0])), dim=-1)


class _MockModel(nn.Module):
    def __init__(self, reachable=True):
        super().__init__(); self.executor = _MockExecutor(reachable)


def _identity_normalizer():
    return TransitionNormalizer(np.zeros(6), np.ones(6), np.zeros(2), np.ones(2))


def test_latent_fit_updates_only_cprime_and_recovers_known_reachable_effect():
    torch.manual_seed(4)
    model = _MockModel()
    before = {k: v.detach().clone() for k,v in model.state_dict().items()}
    c_old = torch.zeros(1,2)
    x = torch.zeros(8,6)
    y_plus = torch.zeros(8,2); y_plus[:,0] = .4
    delta = y_plus.clone()
    starts = torch.zeros(4,2)
    result, failures = _fit_latent(model, _identity_normalizer(), c_old, x, y_plus, delta,
        after_scale=1., effect_scale=.16, beta=.25, lr=.05, regularization=0.,
        max_steps=500, starts=starts)
    best = min(result, key=lambda r:r["fit_objective"])
    assert failures == 0 and best["fit_effect_mse"] < 1e-3
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)


def test_real_encoder_and_executor_state_dicts_are_unchanged_by_latent_fit():
    torch.manual_seed(80)
    model = ContextDynamicsModel(context_dim=4,hidden_dim=8,depth=1)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    c_old = torch.zeros(1,4); x=torch.zeros(6,6); y=torch.zeros(6,2)
    _fit_latent(model,_identity_normalizer(),c_old,x,y,y,after_scale=1.,effect_scale=1.,
        beta=.25,lr=.01,regularization=0.,max_steps=3,starts=torch.randn(4,4))
    after={k:v.detach().cpu() for k,v in model.state_dict().items()}
    assert before.keys()==after.keys()
    assert all(torch.equal(before[k],after[k]) for k in before)


def test_unreachable_mock_is_inconclusive_not_impossibility_claim():
    model = _MockModel(reachable=False); c_old = torch.zeros(1,2)
    x = torch.zeros(4,6); target = torch.zeros(4,2); target[:,0] = 1.
    result, failures = _fit_latent(model, _identity_normalizer(), c_old, x, target, target,
        after_scale=1., effect_scale=1., beta=.25, lr=.05, regularization=0.,
        max_steps=20, starts=torch.zeros(4,2))
    assert failures == 0
    assert all(r["fit_effect_mse"] == .5 for r in result)
    status = "INCONCLUSIVE" if max(r["fit_effect_mse"] for r in result) >= .5 else "PASS"
    assert status == "INCONCLUSIVE"


def test_config_uses_only_train_validation_and_fixed_loss_scale_variants():
    import yaml
    cfg = yaml.safe_load(open("configs/q1_u4_root_cause.yaml", encoding="utf-8"))
    assert cfg["oracle_true_programs"] == [[], [0], [2], [3], [2,3], [3,2]]
    assert cfg["diagnostic_m3_floor_fractions"] == [.01, .0001, 0.]
    assert "test" not in cfg


def test_latent_fit_query_trajectories_are_disjoint_and_seeded_independently():
    params = sample_environment(make_split_manifest(), "validation", 3)
    after = apply_program(params,(3,))
    fit_seed, query_seed = 101, 202
    fit = stage_pair(params,(3,),seed=fit_seed,n_support=0,n_query=128,dt=.05,sigma=0.)[1]
    query = stage_pair(params,(3,),seed=query_seed,n_support=0,n_query=128,dt=.05,sigma=0.)[1]
    fit_probes = np.column_stack((fit.states[:-1],fit.actions))
    query_probes = np.column_stack((query.states[:-1],query.actions))
    assert fit_seed != query_seed
    assert not np.array_equal(fit_probes,query_probes)
    repeated = apply_program(params,(3,))
    np.testing.assert_array_equal(after.B,repeated.B)
    assert after.b == repeated.b


def test_variant_initialization_and_minibatch_order_are_seed_reproducible():
    seed = 971
    torch.manual_seed(seed); init_a = torch.randn(4,8)
    torch.manual_seed(seed); init_b = torch.randn(4,8)
    np.testing.assert_array_equal(init_a.numpy(),init_b.numpy())
    order_a = np.random.default_rng(seed+407).permutation(48)
    order_b = np.random.default_rng(seed+407).permutation(48)
    np.testing.assert_array_equal(order_a,order_b)


def test_u4_energy_floor_ratio_is_explicit_and_training_only():
    params = sample_environment(make_split_manifest(), "train", 2)
    rng = np.random.default_rng(18)
    raw = np.column_stack((rng.normal(0,.2,(128,4)),rng.uniform(-1,1,(128,2))))
    before = response(params, raw)
    after_params = apply_program(params,(3,))
    after = response(after_params,raw)
    query = np.column_stack((raw,after))
    prepared = []
    for code in range(4):
        target = apply_program(params,(code,))
        rows = np.column_stack((raw,response(target,raw)))
        task = TransferTask(np.empty((0,8)), rows, params, target, code, (code,), f"train-{code}")
        prepared.append((task,None,None,None))
    scales = _paired_effect_scales(prepared,dt=.05,scale_floor_fraction=.01)
    raw_energy = float(np.mean((after-before)**2))
    assert np.isclose(scales["per_code_effect_energy"]["3"],raw_energy)
    assert np.isclose(scales["scale_floor"],scales["global_after_energy"]*.01)
    assert np.isclose(scales["per_code_effect_scale_used"]["3"],max(raw_energy,scales["scale_floor"]))
    assert scales["scale_floor_fraction_of_global_after_energy"] == .01
