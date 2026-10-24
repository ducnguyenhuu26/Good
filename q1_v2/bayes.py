"""Stable Gaussian coefficient inference and predictive scoring."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class GaussianPosterior:
    mean: torch.Tensor
    covariance: torch.Tensor
    jitter: float = 0.0
    precision_condition_number: float | None = None


def _chol(matrix: torch.Tensor, *, max_tries: int = 8) -> tuple[torch.Tensor, float]:
    mat = 0.5 * (matrix + matrix.T)
    eye = torch.eye(mat.shape[-1], dtype=mat.dtype, device=mat.device)
    jitter = 0.0
    scale = max(float(torch.diagonal(mat).abs().mean().detach()), 1.0)
    for attempt in range(max_tries):
        factor, info = torch.linalg.cholesky_ex(mat + jitter * eye)
        if int(info.max()) == 0 and torch.isfinite(factor).all():
            return factor, jitter
        jitter = scale * (1e-12 if attempt == 0 else jitter / scale * 10.0)
    raise torch.linalg.LinAlgError("positive-definite Cholesky failed after jitter retries")


def gaussian_posterior(phi: torch.Tensor, y: torch.Tensor, prior_mean: torch.Tensor,
                       prior_cov: torch.Tensor, sigma2: float | torch.Tensor) -> GaussianPosterior:
    """Posterior for stacked 2-row basis observations using KxK Cholesky solves."""
    x = phi.reshape(-1, phi.shape[-1])
    target = y.reshape(-1)
    if x.shape[0] != target.numel():
        raise ValueError("design and target sizes do not align")
    cov = 0.5 * (prior_cov + prior_cov.T)
    l0, jitter0 = _chol(cov)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    precision0 = torch.cholesky_solve(eye, l0)
    noise = torch.as_tensor(sigma2, dtype=x.dtype, device=x.device)
    precision = precision0 + (x.T @ x) / noise
    condition = torch.linalg.cond(precision.detach())
    if not torch.isfinite(condition):
        raise FloatingPointError("non-finite posterior precision condition number")
    lp, jitterp = _chol(precision)
    post_cov = torch.cholesky_solve(eye, lp)
    rhs = precision0 @ prior_mean + x.T @ target / noise
    mean = torch.cholesky_solve(rhs[:, None], lp).squeeze(-1)
    if not (torch.isfinite(mean).all() and torch.isfinite(post_cov).all()):
        raise FloatingPointError("non-finite Gaussian posterior")
    return GaussianPosterior(mean, 0.5 * (post_cov + post_cov.T), max(jitter0, jitterp),
                             float(condition))


def marginal_nll(phi: torch.Tensor, y: torch.Tensor, posterior: GaussianPosterior,
                 sigma2: float | torch.Tensor) -> tuple[torch.Tensor, float]:
    """Exact Gaussian marginal NLL, Cholesky on observation space."""
    x = phi.reshape(-1, phi.shape[-1])
    target = y.reshape(-1)
    # Use the algebraically exact K-space Woodbury form when it is cheaper. Point-estimate
    # covariances are singular by construction and stay on the observation-space path.
    if x.shape[0] > 64 and float(torch.trace(posterior.covariance).detach()) > 1e-14:
        return marginal_nll_woodbury(phi, y, posterior, sigma2), posterior.jitter
    residual = target - x @ posterior.mean
    noise = torch.as_tensor(sigma2, dtype=x.dtype, device=x.device)
    if float(torch.trace(posterior.covariance).detach()) <= 1e-14:
        value = 0.5 * (residual.square().sum() / noise + residual.numel() * torch.log(noise)
                       + residual.numel() * math.log(2 * math.pi))
        return value, posterior.jitter
    c = noise * torch.eye(x.shape[0], dtype=x.dtype, device=x.device) + x @ posterior.covariance @ x.T
    lc, jitter = _chol(c)
    white = torch.linalg.solve_triangular(lc, residual[:, None], upper=False)
    logdet = 2 * torch.log(torch.diagonal(lc)).sum()
    value = 0.5 * (white.square().sum() + logdet + x.shape[0] * math.log(2 * math.pi))
    return value, max(posterior.jitter, jitter)


def marginal_nll_woodbury(phi: torch.Tensor, y: torch.Tensor, posterior: GaussianPosterior,
                          sigma2: float | torch.Tensor) -> torch.Tensor:
    """Equivalent K-space score; useful when observations greatly exceed K."""
    x = phi.reshape(-1, phi.shape[-1])
    target = y.reshape(-1)
    noise = torch.as_tensor(sigma2, dtype=x.dtype, device=x.device)
    prior = posterior.covariance
    lp, _ = _chol(prior)
    eye = torch.eye(prior.shape[0], dtype=x.dtype, device=x.device)
    p = torch.cholesky_solve(eye, lp) + x.T @ x / noise
    l, _ = _chol(p)
    r = target - x @ posterior.mean
    xr = x.T @ r / noise
    quad = (r @ r) / noise - xr @ torch.cholesky_solve(xr[:, None], l).squeeze(-1)
    logdet = x.shape[0] * torch.log(noise) + 2 * torch.log(torch.diagonal(l)).sum() + 2 * torch.log(torch.diagonal(lp)).sum()
    return 0.5 * (quad + logdet + x.shape[0] * math.log(2 * math.pi))
