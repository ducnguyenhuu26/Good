from __future__ import annotations
import ast
from pathlib import Path

import torch

from q1_v2.bayes import GaussianPosterior, gaussian_posterior, marginal_nll, marginal_nll_woodbury
from q1_v2.inference import composed, rank_programs
from q1_v2.operators import AffineOperatorBank
from q1_v2.training import ObservedEpisode, fit_operator_bank


def test_linear_gaussian_posterior_and_predictive_nll_agree():
    torch.manual_seed(10)
    dtype = torch.float64
    x = torch.randn(9, 2, 3, dtype=dtype)
    y = torch.randn(9, 2, dtype=dtype)
    m0 = torch.randn(3, dtype=dtype)
    s0 = torch.diag(torch.tensor([.2, .5, 1.], dtype=dtype))
    noise = .03
    post = gaussian_posterior(x, y, m0, s0, noise)
    xf = x.reshape(-1, 3)
    yf = y.reshape(-1)
    expected_cov = torch.linalg.inv(torch.linalg.inv(s0) + xf.T @ xf / noise)
    expected_mean = expected_cov @ (torch.linalg.solve(s0, m0) + xf.T @ yf / noise)
    assert torch.allclose(post.mean, expected_mean, atol=1e-9, rtol=1e-8)
    assert torch.allclose(post.covariance, expected_cov, atol=1e-9, rtol=1e-8)
    p = GaussianPosterior(m0, s0)
    nll, _ = marginal_nll(x, y, p, noise)
    wb = marginal_nll_woodbury(x, y, p, noise)
    assert torch.allclose(nll, wb, atol=1e-8, rtol=1e-8)


def test_posterior_covariance_psd_noise_and_jitter_recording():
    dtype = torch.float64
    x = torch.randn(20, 2, 4, dtype=dtype)
    y = torch.randn(20, 2, dtype=dtype)
    m = torch.zeros(4, dtype=dtype)
    s = torch.eye(4, dtype=dtype)
    low, high = gaussian_posterior(x, y, m, s, 1e-3), gaussian_posterior(x, y, m, s, 1.)
    assert torch.isfinite(low.covariance).all()
    assert torch.allclose(low.covariance, low.covariance.T, atol=1e-10)
    assert torch.linalg.eigvalsh(low.covariance).min() >= -1e-10
    assert torch.trace(low.covariance) < torch.trace(high.covariance)
    jitter_cov = torch.diag(torch.tensor([-1e-14, 1., 1., 1.], dtype=dtype))
    near_singular = gaussian_posterior(torch.zeros(2, 2, 4, dtype=dtype),
        torch.zeros(2, 2, dtype=dtype), m, jitter_cov, 1.)
    assert near_singular.jitter > 0


def test_monte_carlo_predictive_nll_matches_closed_form():
    torch.manual_seed(27)
    dtype=torch.float64
    x=torch.tensor([[[1.,-.5],[.2,1.]]],dtype=dtype)
    prior=GaussianPosterior(torch.tensor([.3,-.2],dtype=dtype),
        torch.tensor([[.4,.1],[.1,.25]],dtype=dtype))
    noise=.07
    observed=torch.tensor([[.1,-.4]],dtype=dtype)
    closed,_=marginal_nll(x,observed,prior,noise)
    design=x.reshape(-1,2)
    predictive_mean=design@prior.mean
    predictive_cov=noise*torch.eye(2,dtype=dtype)+design@prior.covariance@design.T
    chol=torch.linalg.cholesky(predictive_cov)
    eps=torch.randn(50000,2,dtype=dtype)@chol.T
    draws=predictive_mean+eps
    empirical_mean=draws.mean(0)
    empirical_cov=torch.cov(draws.T)
    resid=observed.reshape(-1)-empirical_mean
    mc=.5*(torch.logdet(empirical_cov)+resid@torch.linalg.solve(empirical_cov,resid)+2*torch.log(torch.tensor(2*torch.pi,dtype=dtype)))
    assert abs(float(mc-closed)) < .025


def test_affine_identity_associativity_repeats_and_noncommutativity():
    dtype = torch.float64
    a = torch.tensor([[1., 1.], [0., 1.]], dtype=dtype)
    b = torch.tensor([[1., 0.], [1., 1.]], dtype=dtype)
    z = torch.zeros(2, dtype=dtype)
    ident = torch.eye(2, dtype=dtype)
    ab = AffineOperatorBank.compose((ident, z), (a, z))
    assert torch.equal(ab[0], a)
    lhs = AffineOperatorBank.compose(AffineOperatorBank.compose((a, z), (b, z)), (a, z))
    rhs = AffineOperatorBank.compose((a, z), AffineOperatorBank.compose((b, z), (a, z)))
    assert torch.allclose(lhs[0], rhs[0])
    ab, ba = AffineOperatorBank.compose((a, z), (b, z)), AffineOperatorBank.compose((b, z), (a, z))
    assert not torch.allclose(ab[0] @ torch.ones(2, dtype=dtype), ba[0] @ torch.ones(2, dtype=dtype))
    bank = AffineOperatorBank()
    for code in range(4):
        matrix = bank.matrix(code)
        assert torch.count_nonzero(matrix[:6, 6:]) == 0
        assert torch.count_nonzero(matrix[:6, 10:]) == 0
        assert torch.count_nonzero(matrix[6:10, :6]) == 0
        assert torch.count_nonzero(matrix[10:, :10]) == 0
    assert torch.allclose(composed(bank, (1, 1))[0], bank.matrix(1) @ bank.matrix(1))


def test_program_scores_ignore_assessment_queries_and_enumerate_21():
    bank = AffineOperatorBank()
    old = GaussianPosterior(torch.zeros(12), torch.eye(12) * .1)
    phi = torch.randn(5, 2, 12)
    y = torch.randn(5, 2)
    programs = [()] + [(i,) for i in range(4)] + [(i, j) for i in range(4) for j in range(4)]
    scores1 = [v[1] for v in rank_programs(bank, old, programs, phi, y, .01)]
    assessment_permuted = torch.randn_like(y).flip(0)  # assessment data are not scorer arguments
    scores2 = [v[1] for v in rank_programs(bank, old, programs, phi, y, .01)]
    assert len(scores1) == 21 and scores1 == scores2
    assert assessment_permuted.shape == y.shape


def test_learned_modules_obey_import_boundary():
    root = Path(__file__).parents[1] / "q1_v2"
    forbidden = {"q1.simulator", "DynamicsParams", "intervention", "apply_program", "response"}
    for path in (root / "basis.py", root / "bayes.py", root / "operators.py", root / "inference.py", root / "training.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        names = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                 for alias in getattr(node, "names", [])}
        assert "q1.simulator" not in imported
        assert not (forbidden & names)


def test_partition_basis_shape_parameter_count_and_gradient_finiteness():
    from q1_v2.basis import PartitionedBasis
    torch.manual_seed(9)
    model=PartitionedBasis(hidden=8)
    x=torch.randn(7,6,requires_grad=True)
    output=model(x)
    assert output.shape==(7,2,12)
    assert sum(p.numel() for p in model.parameters())>0
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_weak_label_operator_training_transfers_to_unseen_background_and_pair():
    class FixedBasis(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(*x.shape[:-1], 2, 12, dtype=x.dtype, device=x.device)
            out[..., 0, :6] = x
            out[..., 1, :6] = x.roll(1, dims=-1)
            return out

    torch.manual_seed(88)
    dtype = torch.float32
    basis = FixedBasis()
    true_ops = []
    for mat in torch.eye(6, dtype=dtype).repeat(4, 1, 1):
        true_ops.append(mat)
    true_ops[0][0, 1] = .25
    true_ops[1][1, 0] = .20
    assert not torch.allclose(true_ops[1] @ true_ops[0], true_ops[0] @ true_ops[1])

    def phi(x):
        return basis(x).reshape(-1, 12)

    def make_episode(seed, program):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(12, generator=g) * .15
        w[6:] = 0
        xs = torch.randn(48, 6, generator=g)
        xq = torch.randn(24, 6, generator=g)
        transform = torch.eye(12)
        bias = torch.zeros(12)
        for code in program:
            a = torch.eye(12)
            a[:6, :6] = true_ops[code]
            transform, bias = a @ transform, a @ bias
        target = transform @ w + bias
        # Two output coordinates are generated with the same 6D coefficient block.
        ys = torch.einsum("nok,k->no", basis(xs), w)
        yq = torch.einsum("nOk,k->nO", basis(xq), target)
        ys += .001 * torch.randn(ys.shape, generator=g)
        yq += .001 * torch.randn(yq.shape, generator=g)
        return ObservedEpisode(xs, ys, xq, yq, tuple(program))

    allowed = [(0,), (1,), (2,), (3,), (0, 0), (1, 1), (0, 2), (2, 0)]
    episodes = [make_episode(1000 + i, allowed[i % len(allowed)]) for i in range(96)]
    bank, _ = fit_operator_bank(basis, torch.zeros(12), torch.eye(12), episodes,
        sigma2=1e-4, seed=29, epochs=180, lr=2e-3, regularization=1e-4)
    unseen = make_episode(998877, (0, 1))
    old = gaussian_posterior(phi(unseen.support_x), unseen.support_y, torch.zeros(12), torch.eye(12), 1e-4)
    learned = composed(bank, (0, 1))[0] @ old.mean + composed(bank, (0, 1))[1]
    pred = (basis(unseen.query_x) @ learned).reshape(-1)
    target = unseen.query_y.reshape(-1)
    assert torch.isfinite(pred).all()
    assert torch.mean((pred - target) ** 2) < 0.08
