"""Weak-label training APIs. Inputs are observed transition episodes only."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.optim import Adam

from q1_v2.bayes import GaussianPosterior, marginal_nll
from q1_v2.operators import AffineOperatorBank


@dataclass
class ObservedEpisode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    opaque_program: tuple[int, ...] = ()


def design(basis: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Inputs [n,6] -> design [2n,K], preserving x/y output row ordering.
    values = basis(x)
    return values.reshape(-1, values.shape[-1])


def balanced_episode_order(episodes: list[ObservedEpisode], *, seed: int) -> list[int]:
    """Return a seeded full-coverage order, round-robin balanced by opaque code.

    Each episode appears exactly once. For pairs, it is queued under both of its
    opaque codes and contributes to both running exposure counts. The next episode
    is taken from an available code with the lowest current exposure count.
    """
    if not episodes:
        raise ValueError("operator training requires at least one episode")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    labels = [tuple(sorted(set(int(c) for c in ep.opaque_program))) for ep in episodes]
    code_values = sorted({c for row in labels for c in row})
    if not code_values:
        code_values = [-1]  # empty-program-only synthetic/control datasets
    queues: dict[int, list[int]] = {c: [] for c in code_values}
    for idx, codes in enumerate(labels):
        membership = codes or (-1,)
        for code in membership:
            queues.setdefault(code, []).append(idx)
    for code, queue in queues.items():
        if queue:
            permutation = torch.randperm(len(queue), generator=generator).tolist()
            queues[code] = [queue[i] for i in permutation]

    positions = {code: 0 for code in queues}
    exposure = {code: 0 for code in queues}
    remaining = set(range(len(episodes)))
    result: list[int] = []
    while remaining:
        available = [code for code, queue in queues.items()
                     if any(i in remaining for i in queue[positions[code]:])]
        if not available:
            raise RuntimeError("balanced sampler lost unvisited episodes")
        code = min(available, key=lambda c: (exposure[c],
            float(torch.rand((), generator=generator))))
        queue = queues[code]
        while positions[code] < len(queue) and queue[positions[code]] not in remaining:
            positions[code] += 1
        if positions[code] >= len(queue):
            continue
        idx = queue[positions[code]]
        positions[code] += 1
        remaining.remove(idx)
        result.append(idx)
        membership = labels[idx] or (-1,)
        for member in membership:
            exposure[member] += 1
    return result


def fit_basis(basis: nn.Module, episodes: list[ObservedEpisode], *, sigma2: float,
              seed: int, epochs: int = 18, batch_size: int = 8, lr: float = 1e-3,
              inner_episodes: list[ObservedEpisode] | None = None) -> dict:
    """Pretrain Phi and train-only Gaussian prior via support/query predictive NLL."""
    torch.manual_seed(seed)
    dtype = next(basis.parameters()).dtype
    device = next(basis.parameters()).device
    k = 12
    prior_mean = nn.Parameter(torch.zeros(k, dtype=dtype, device=device))
    log_variance = nn.Parameter(torch.zeros(k, dtype=dtype, device=device))
    optimizer = Adam([*basis.parameters(), prior_mean, log_variance], lr=lr)
    curves = []
    best = None
    best_loss = float("inf")
    rng = torch.Generator().manual_seed(seed + 59)
    for epoch in range(epochs):
        basis.train()
        order = torch.randperm(len(episodes), generator=rng).tolist()
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [episodes[i] for i in order[start:start + batch_size]]
            # Batch tasks, with independent observed support and query sequences.
            sx = torch.stack([e.support_x for e in batch]).to(device=device, dtype=dtype)
            sy = torch.stack([e.support_y for e in batch]).to(device=device, dtype=dtype)
            qx = torch.stack([e.query_x for e in batch]).to(device=device, dtype=dtype)
            qy = torch.stack([e.query_y for e in batch]).to(device=device, dtype=dtype)
            ps = basis(sx).permute(0, 2, 1, 3).reshape(len(batch), -1, k)
            py = sy.permute(0, 2, 1).reshape(len(batch), -1)
            pq = basis(qx).permute(0, 2, 1, 3).reshape(len(batch), -1, k)
            qtarget = qy.permute(0, 2, 1).reshape(len(batch), -1)
            s0 = torch.diag_embed(torch.exp(log_variance.clamp(-12, 12)).expand(len(batch), -1))
            eye = torch.eye(k, dtype=dtype, device=device).expand(len(batch), -1, -1)
            p0 = torch.diag_embed(torch.exp(-log_variance.clamp(-12, 12)).expand(len(batch), -1))
            precision = p0 + ps.transpose(1, 2) @ ps / sigma2
            chol = torch.linalg.cholesky(precision)
            cov = torch.cholesky_inverse(chol)
            rhs = (p0 @ prior_mean.expand(len(batch), -1).unsqueeze(-1)
                   + ps.transpose(1, 2) @ py.unsqueeze(-1) / sigma2)
            mean = torch.cholesky_solve(rhs, chol).squeeze(-1)
            residual = qtarget - (pq @ mean.unsqueeze(-1)).squeeze(-1)
            obs_cov = sigma2 * torch.eye(pq.shape[1], dtype=dtype, device=device).expand(len(batch), -1, -1)
            obs_cov = obs_cov + pq @ cov @ pq.transpose(1, 2)
            l = torch.linalg.cholesky(0.5 * (obs_cov + obs_cov.transpose(1, 2)))
            white = torch.linalg.solve_triangular(l, residual.unsqueeze(-1), upper=False)
            nll = 0.5 * (white.square().sum(dim=(1, 2))
                         + 2 * torch.log(torch.diagonal(l, dim1=-2, dim2=-1)).sum(-1)
                         + pq.shape[1] * math.log(2 * math.pi)).mean()
            if not torch.isfinite(nll):
                raise FloatingPointError("non-finite basis training loss")
            optimizer.zero_grad(set_to_none=True)
            nll.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in basis.parameters()):
                raise FloatingPointError("non-finite basis gradient")
            torch.nn.utils.clip_grad_norm_([*basis.parameters(), prior_mean, log_variance], 5.0)
            optimizer.step()
            losses.append(float(nll.detach()))
        validation = evaluate_basis_nll(basis, inner_episodes or episodes,
                                        prior_mean.detach(), log_variance.detach(), sigma2)
        curves.append({"epoch": epoch + 1, "train_nll": sum(losses) / max(len(losses), 1),
                       "inner_nll": validation})
        if validation < best_loss:
            best_loss = validation
            best = ({k: v.detach().clone() for k, v in basis.state_dict().items()},
                    prior_mean.detach().clone(), log_variance.detach().clone(), epoch + 1)
    if best is not None:
        basis.load_state_dict(best[0])
        prior_mean.data.copy_(best[1])
        log_variance.data.copy_(best[2])
    return {"prior_mean": prior_mean.detach(), "prior_covariance": torch.diag(torch.exp(log_variance.detach())),
            "sigma2": float(sigma2), "curve": curves, "selected_epoch": best[3] if best else epochs}


def evaluate_basis_nll(basis, episodes, prior_mean, log_variance, sigma2):
    losses = []
    basis.eval()
    with torch.no_grad():
        cov0 = torch.diag(torch.exp(log_variance))
        for episode in episodes:
            xs = design(basis, episode.support_x)
            xq = design(basis, episode.query_x)
            from q1_v2.bayes import gaussian_posterior
            post = gaussian_posterior(xs, episode.support_y, prior_mean, cov0, sigma2)
            value, _ = marginal_nll(xq, episode.query_y, post, sigma2)
            losses.append(float(value))
    return sum(losses) / max(1, len(losses))


def fit_operator_bank(basis: nn.Module, prior_mean: torch.Tensor, prior_cov: torch.Tensor,
                      episodes: list[ObservedEpisode], *, sigma2: float, seed: int,
                      epochs: int = 160, lr: float = 1e-3, regularization: float = 1e-3,
                      inner_episodes: list[ObservedEpisode] | None = None,
                      length_penalty: float = 0.0,
                      use_old_support: bool = True) -> tuple[AffineOperatorBank, dict]:
    """Learn same-parameterization affine operators from opaque observed episodes."""
    from q1_v2.bayes import gaussian_posterior
    if not episodes:
        raise ValueError("operator training requires nonempty training episodes")
    if inner_episodes is None or not inner_episodes:
        raise ValueError("operator checkpoint selection requires nonempty inner-training episodes")
    validation_episodes = inner_episodes
    basis.eval()
    for parameter in basis.parameters():
        parameter.requires_grad_(False)
    bank = AffineOperatorBank().to(device=prior_mean.device, dtype=prior_mean.dtype)
    optimizer = Adam(bank.parameters(), lr=lr)
    curves = []
    best_state = None
    best_inner_nll = float("inf")
    best_epoch = 0
    epoch_coverage = []
    seed_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for epoch in range(epochs):
        bank.train()
        order_seed = int(torch.randint(0, 2**31 - 1, (), generator=seed_generator))
        order = balanced_episode_order(episodes, seed=order_seed)
        train_losses, penalties, jitters = [], [], []
        seen_programs: dict[str, int] = {}
        code_counts = {str(c): 0 for ep in episodes for c in ep.opaque_program}
        for episode_index in order:
            ep = episodes[episode_index]
            xold = design(basis, ep.support_x)
            old = (gaussian_posterior(xold, ep.support_y, prior_mean, prior_cov, sigma2)
                   if use_old_support else GaussianPosterior(prior_mean, prior_cov))
            # Opaque labels are supplied by the sampler; no physical metadata enters this API.
            a, b = AffineOperatorBank.identity(12, dtype=prior_mean.dtype, device=prior_mean.device)
            for code in ep.opaque_program:
                a, b = AffineOperatorBank.compose((a, b), (bank.matrix(int(code)), bank.bias[int(code)]))
            transformed = GaussianPosterior(a @ old.mean + b, a @ old.covariance @ a.T, old.jitter)
            xnew = design(basis, ep.query_x)
            loss, jitter = marginal_nll(xnew, ep.query_y, transformed, sigma2)
            penalty = regularization * bank.regularization()
            total = loss + penalty + float(length_penalty) * len(ep.opaque_program)
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite operator training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in bank.parameters()):
                raise FloatingPointError("non-finite operator gradient")
            torch.nn.utils.clip_grad_norm_(bank.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
            penalties.append(float(penalty.detach()))
            jitters.append(float(jitter))
            program_key = ",".join(map(str, ep.opaque_program)) or "empty"
            seen_programs[program_key] = seen_programs.get(program_key, 0) + 1
            for code in ep.opaque_program:
                code_counts[str(code)] = code_counts.get(str(code), 0) + 1

        inner_nll = evaluate_operator_nll(basis, bank, prior_mean, prior_cov,
            validation_episodes, sigma2, use_old_support=use_old_support)
        coverage = {"epoch": epoch + 1, "optimizer_seed": int(seed),
            "training_episode_count": len(order), "unique_training_episodes": len(set(order)),
            "program_episode_counts": seen_programs, "opaque_code_episode_exposures": code_counts,
            "shuffle_seed": order_seed}
        epoch_coverage.append(coverage)
        curves.append({"epoch": epoch + 1, "training_episode_count": len(order),
            "train_nll": float(sum(train_losses) / len(train_losses)),
            "regularization": float(sum(penalties) / len(penalties)),
            "max_jitter": max(jitters, default=0.0), "inner_nll": inner_nll,
            "selected": inner_nll < best_inner_nll})
        if inner_nll < best_inner_nll:
            best_inner_nll = inner_nll
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in bank.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no finite inner-validation checkpoint was selected")
    bank.load_state_dict(best_state)
    return bank, {"curve": curves, "epoch_coverage": epoch_coverage,
                  "regularization": regularization, "inner_nll": best_inner_nll,
                  "best_epoch": best_epoch, "seed": int(seed),
                  "selection_split": "inner-train backgrounds only",
                  "training_episode_count_per_epoch": len(episodes),
                  "uses_old_support": use_old_support}


def evaluate_operator_nll(basis, bank, prior_mean, prior_cov, episodes, sigma2,
                          *, use_old_support: bool = True):
    from q1_v2.bayes import gaussian_posterior
    values = []
    with torch.no_grad():
        for ep in episodes:
            old = (gaussian_posterior(design(basis, ep.support_x), ep.support_y,
                                      prior_mean, prior_cov, sigma2) if use_old_support
                   else GaussianPosterior(prior_mean, prior_cov))
            a, b = AffineOperatorBank.identity(12, dtype=prior_mean.dtype, device=prior_mean.device)
            for code in ep.opaque_program:
                a, b = AffineOperatorBank.compose((a, b), (bank.matrix(int(code)), bank.bias[int(code)]))
            p = GaussianPosterior(a @ old.mean + b, a @ old.covariance @ a.T)
            nll, _ = marginal_nll(design(basis, ep.query_x), ep.query_y, p, sigma2)
            values.append(float(nll))
    return sum(values) / max(1, len(values))
