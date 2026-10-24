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


@dataclass
class ObservedEffectTask:
    """Cross-fitted, noisy-observation effect target; contains no simulator metadata."""
    opaque_code: int
    before_coeff: torch.Tensor
    delta_coeff: torch.Tensor
    probe_x: torch.Tensor
    background_id: str = ""
    fold: int = 0
    noise_variance: float = 0.0


def design(basis: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Inputs [n,6] -> design [2n,K], preserving x/y output row ordering.
    values = basis(x)
    return values.reshape(-1, values.shape[-1])


def _batched_cholesky(matrix: torch.Tensor, *, label: str) -> tuple[torch.Tensor,float]:
    """Numerically stable batched Cholesky with trace-scaled diagonal jitter."""
    mat=0.5*(matrix+matrix.transpose(-1,-2))
    chol,info=torch.linalg.cholesky_ex(mat)
    if int(info.max())==0 and torch.isfinite(chol).all():return chol,0.0
    batch=mat.shape[0];k=mat.shape[-1]
    eye=torch.eye(k,dtype=mat.dtype,device=mat.device).expand(batch,-1,-1)
    scale=torch.diagonal(mat,dim1=-2,dim2=-1).abs().mean(-1).clamp_min(1.0)
    for exponent in range(-8,-1):
        jitter=scale*(10.0**exponent)
        chol,info=torch.linalg.cholesky_ex(mat+jitter[:,None,None]*eye)
        if int(info.max())==0 and torch.isfinite(chol).all():return chol,float(jitter.max().detach())
    raise torch.linalg.LinAlgError(f"{label} remained non-PD after trace-scaled jitter retries")


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
    k = int(sum(basis.widths)) if hasattr(basis, "widths") else 12
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
        max_cholesky_jitter=0.0
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
            precision = 0.5 * (precision + precision.transpose(1, 2))
            chol,precision_jitter=_batched_cholesky(precision,label='basis training precision')
            max_cholesky_jitter=max(max_cholesky_jitter,precision_jitter)
            cov = torch.cholesky_inverse(chol)
            rhs = (p0 @ prior_mean.expand(len(batch), -1).unsqueeze(-1)
                   + ps.transpose(1, 2) @ py.unsqueeze(-1) / sigma2)
            mean = torch.cholesky_solve(rhs, chol).squeeze(-1)
            residual = qtarget - (pq @ mean.unsqueeze(-1)).squeeze(-1)
            obs_cov = sigma2 * torch.eye(pq.shape[1], dtype=dtype, device=device).expand(len(batch), -1, -1)
            obs_cov = obs_cov + pq @ cov @ pq.transpose(1, 2)
            l,obs_jitter=_batched_cholesky(obs_cov,label='basis predictive covariance')
            max_cholesky_jitter=max(max_cholesky_jitter,obs_jitter)
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
                       "max_cholesky_jitter":max_cholesky_jitter,
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
                      use_old_support: bool = True,
                      normalize_nll_per_observation: bool = False) -> tuple[AffineOperatorBank, dict]:
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
    widths = tuple(getattr(basis, "widths", (6, 4, 2)))
    if sum(widths) != prior_mean.numel():
        raise ValueError("basis coefficient width must match the Gaussian prior dimension")
    bank = AffineOperatorBank(widths=widths).to(device=prior_mean.device, dtype=prior_mean.dtype)
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
            a, b = AffineOperatorBank.identity(prior_mean.numel(), dtype=prior_mean.dtype, device=prior_mean.device)
            for code in ep.opaque_program:
                a, b = AffineOperatorBank.compose((a, b), (bank.matrix(int(code)), bank.bias[int(code)]))
            transformed = GaussianPosterior(a @ old.mean + b, a @ old.covariance @ a.T, old.jitter)
            xnew = design(basis, ep.query_x)
            loss, jitter = marginal_nll(xnew, ep.query_y, transformed, sigma2)
            penalty = regularization * bank.regularization()
            nll_term=loss/max(1,ep.query_y.numel()) if normalize_nll_per_observation else loss
            total = nll_term + penalty + float(length_penalty) * len(ep.opaque_program)
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
                  "nll_normalized_per_observation":bool(normalize_nll_per_observation),
                  "uses_old_support": use_old_support}


def evaluate_effect_tasks(basis: nn.Module, bank: AffineOperatorBank,
                          tasks: list[ObservedEffectTask], normalizers: dict[int,float]) -> dict:
    """Return per-code cross-fitted effect skill, with background-level inputs retained."""
    outputs={}
    k=sum(bank.widths)
    with torch.no_grad():
        for task in tasks:
            code=int(task.opaque_code)
            a=bank.matrix(code);b=bank.bias[code]
            w=task.before_coeff.to(dtype=a.dtype,device=a.device)
            target_delta=task.delta_coeff.to(dtype=a.dtype,device=a.device)
            phi=design(basis,task.probe_x.to(dtype=a.dtype,device=a.device))
            eye=torch.eye(k,dtype=a.dtype,device=a.device)
            pred=phi@((a-eye)@w+b)
            target=phi@target_delta
            err=float(torch.mean((pred-target).square()))
            energy=float(torch.mean(target.square()))
            outputs.setdefault(code,[]).append({'background_id':task.background_id,
                'fold':task.fold,'mse':err,'target_energy':energy,
                'skill':(1.-err/energy if energy>1e-24 else None),
                'normalized_mse':err/max(float(normalizers[code]),1e-30)})
    result={}
    for code,vals in outputs.items():
        energy_sum=sum(v['target_energy'] for v in vals)
        valid_skills=[v['skill'] for v in vals if v['skill'] is not None]
        result[code]={'skill':(float(1.-sum(v['mse'] for v in vals)/energy_sum) if energy_sum>1e-24 else None),
            'mean_task_skill':(float(sum(valid_skills)/len(valid_skills)) if valid_skills else None),
            'normalized_mse':float(sum(v['normalized_mse'] for v in vals)/len(vals)),
            'tasks':vals}
    return result


def fit_operator_bank_effect_aware(basis: nn.Module, prior_mean: torch.Tensor,
        prior_cov: torch.Tensor, episodes: list[ObservedEpisode],
        effect_tasks: list[ObservedEffectTask], *, effect_scales: dict[int,float],
        sigma2: float, seed: int, epochs: int, lr: float, regularization: float,
        effect_weight: float, inner_episodes: list[ObservedEpisode],
        inner_effect_tasks: list[ObservedEffectTask],
        inner_effect_scales: dict[int,float], length_penalty: float=0.0,
        selection_code_groups: dict[str,int],
        u12_min_skill: float=-0.25) -> tuple[AffineOperatorBank,dict]:
    """Train code-balanced full epochs on after-NLL and cross-fitted observed effects."""
    from q1_v2.bayes import gaussian_posterior
    if not episodes or not effect_tasks or not inner_episodes or not inner_effect_tasks:
        raise ValueError('effect-aware training requires train and inner NLL/effect tasks')
    widths=tuple(getattr(basis,'widths',(6,4,2)))
    if sum(widths)!=prior_mean.numel():raise ValueError('basis and prior dimensions disagree')
    basis.eval()
    for p in basis.parameters():p.requires_grad_(False)
    bank=AffineOperatorBank(widths=widths).to(device=prior_mean.device,dtype=prior_mean.dtype)
    optimizer=Adam(bank.parameters(),lr=lr)
    generator=torch.Generator(device='cpu').manual_seed(int(seed))
    train_codes=sorted({int(t.opaque_code) for t in effect_tasks})
    if set(train_codes)!=set(effect_scales):raise ValueError('train-only effect normalization must cover observed opaque codes')
    # Identity bank is the fixed after-NLL and effect baseline for inner selection.
    identity=AffineOperatorBank(widths=widths).to(device=prior_mean.device,dtype=prior_mean.dtype)
    for blocks in identity.blocks:
        with torch.no_grad():blocks.copy_(torch.eye(blocks.shape[-1],dtype=blocks.dtype,device=blocks.device).expand_as(blocks))
    baseline_nll=evaluate_operator_nll(basis,identity,prior_mean,prior_cov,inner_episodes,sigma2)
    curves=[];coverage=[];best_state=None;best_score=float('-inf');best_epoch=0
    fallback_states=[]
    for epoch in range(epochs):
        bank.train()
        order_seed=int(torch.randint(0,2**31-1,(),generator=generator))
        order=balanced_episode_order(episodes,seed=order_seed)
        nll_values=[]
        for idx in order:
            ep=episodes[idx]
            old=gaussian_posterior(design(basis,ep.support_x),ep.support_y,prior_mean,prior_cov,sigma2)
            a,b=AffineOperatorBank.identity(prior_mean.numel(),dtype=prior_mean.dtype,device=prior_mean.device)
            for code in ep.opaque_program:
                a,b=AffineOperatorBank.compose((a,b),(bank.matrix(int(code)),bank.bias[int(code)]))
            transformed=GaussianPosterior(a@old.mean+b,a@old.covariance@a.T,old.jitter)
            loss,jitter=marginal_nll(design(basis,ep.query_x),ep.query_y,transformed,sigma2)
            normalized=loss/max(1,ep.query_y.numel())
            total=normalized+regularization*bank.regularization()+float(length_penalty)*len(ep.opaque_program)
            optimizer.zero_grad(set_to_none=True);total.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in bank.parameters()):
                raise FloatingPointError('non-finite operator gradient')
            torch.nn.utils.clip_grad_norm_(bank.parameters(),5.0);optimizer.step()
            nll_values.append(float(loss.detach()))
        # Each code has exactly two cross-fit fold tasks per training background.
        effect_labels=[ObservedEpisode(torch.empty(0),torch.empty(0),torch.empty(0),torch.empty(0),
            (int(task.opaque_code),)) for task in effect_tasks]
        effect_seed=int(torch.randint(0,2**31-1,(),generator=generator))
        effect_order=balanced_episode_order(effect_labels,seed=effect_seed)
        code_coverage={str(c):0 for c in train_codes};effect_losses=[];effect_grad_norms={str(c):[] for c in train_codes}
        for ti in effect_order:
            task=effect_tasks[ti];code=int(task.opaque_code);code_coverage[str(code)]+=1
            a=bank.matrix(code);b=bank.bias[code]
            w=task.before_coeff.to(dtype=a.dtype,device=a.device)
            delta=task.delta_coeff.to(dtype=a.dtype,device=a.device)
            phi=design(basis,task.probe_x.to(dtype=a.dtype,device=a.device))
            eye=torch.eye(prior_mean.numel(),dtype=a.dtype,device=a.device)
            prediction=phi@((a-eye)@w+b);target=phi@delta
            effect_loss=(prediction-target).square().mean()/max(float(effect_scales[code]),1e-30)
            total=float(effect_weight)*effect_loss+regularization*bank.regularization()
            optimizer.zero_grad(set_to_none=True);total.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in bank.parameters()):
                raise FloatingPointError('non-finite effect-aware operator gradient')
            grad=torch.nn.utils.clip_grad_norm_(bank.parameters(),5.0)
            effect_grad_norms[str(code)].append(float(grad))
            optimizer.step();effect_losses.append(float(effect_loss.detach()))
        inner_nll=evaluate_operator_nll(basis,bank,prior_mean,prior_cov,inner_episodes,sigma2)
        effect_metrics=evaluate_effect_tasks(basis,bank,inner_effect_tasks,inner_effect_scales)
        code_skill={str(c):effect_metrics[c]['skill'] for c in effect_metrics}
        after_improvement=(baseline_nll-inner_nll)/(abs(baseline_nll)+1.)
        target_codes=[int(selection_code_groups[k]) for k in ('U3','U4') if k in selection_code_groups and int(selection_code_groups[k]) in effect_metrics]
        valid_primary=[effect_metrics[c]['skill'] for c in target_codes if effect_metrics[c]['skill'] is not None]
        primary_effect=float(sum(valid_primary)/len(valid_primary)) if valid_primary else float('-inf')
        guard_codes=[int(selection_code_groups[k]) for k in ('U1','U2') if k in selection_code_groups and int(selection_code_groups[k]) in effect_metrics]
        u12_ok=all(effect_metrics[c]['skill'] is not None and effect_metrics[c]['skill']>=u12_min_skill for c in guard_codes)
        combined=primary_effect+0.1*after_improvement
        selected=(u12_ok and combined>best_score)
        coverage.append({'epoch':epoch+1,'optimizer_seed':int(seed),'nll_episode_count':len(order),
            'unique_nll_episodes':len(set(order)),'effect_task_count':len(effect_order),
            'effect_shuffle_seed':effect_seed,
            'effect_task_counts_by_code':code_coverage,'nll_shuffle_seed':order_seed})
        curves.append({'epoch':epoch+1,'train_nll_per_observation':float(sum(nll_values)/max(1,len(nll_values)))/max(1,episodes[0].query_y.numel()),
            'train_effect_normalized_mse':float(sum(effect_losses)/max(1,len(effect_losses))),'inner_nll':inner_nll,
            'identity_inner_nll':baseline_nll,'inner_after_nll_improvement':after_improvement,
            'inner_effect_skill_by_code':code_skill,'inner_U12_guard_pass':u12_ok,
            'combined_selection_score':combined,'selected':selected,
            'mean_effect_gradient_norm_by_code':{c:float(sum(v)/max(1,len(v))) for c,v in effect_grad_norms.items()}})
        fallback_states.append({k:v.detach().clone() for k,v in bank.state_dict().items()})
        if selected:
            best_score=combined;best_epoch=epoch+1
            best_state={k:v.detach().clone() for k,v in bank.state_dict().items()}
    if best_state is None:
        # Keep the best combined checkpoint if the non-regression guard was never met;
        # status records this and the caller must classify it as a failed selection gate.
        candidates=[(r['combined_selection_score'],i+1) for i,r in enumerate(curves)]
        best_score,best_epoch=max(candidates)
        best_state=fallback_states[best_epoch-1];bank.load_state_dict(best_state)
    else:
        bank.load_state_dict(best_state)
    return bank,{'curve':curves,'epoch_coverage':coverage,'seed':int(seed),
        'effect_weight':float(effect_weight),'regularization':float(regularization),
        'effect_scales_train_only':{str(k):float(v) for k,v in effect_scales.items()},
        'inner_effect_scales_train_inner':{str(k):float(v) for k,v in inner_effect_scales.items()},
        'best_epoch':best_epoch,'best_combined_score':best_score,
        'selection_metric':'mean inner U3/U4 cross-fit effect skill + 0.1 normalized after-NLL improvement; U1/U2 skill guard',
        'u12_guard_passed_any_epoch':any(r['inner_U12_guard_pass'] for r in curves),
        'selection_split':'inner-training backgrounds only'}


def evaluate_operator_nll(basis, bank, prior_mean, prior_cov, episodes, sigma2,
                          *, use_old_support: bool = True):
    from q1_v2.bayes import gaussian_posterior
    values = []
    with torch.no_grad():
        for ep in episodes:
            old = (gaussian_posterior(design(basis, ep.support_x), ep.support_y,
                                      prior_mean, prior_cov, sigma2) if use_old_support
                   else GaussianPosterior(prior_mean, prior_cov))
            a, b = AffineOperatorBank.identity(prior_mean.numel(), dtype=prior_mean.dtype, device=prior_mean.device)
            for code in ep.opaque_program:
                a, b = AffineOperatorBank.compose((a, b), (bank.matrix(int(code)), bank.bias[int(code)]))
            p = GaussianPosterior(a @ old.mean + b, a @ old.covariance @ a.T)
            nll, _ = marginal_nll(design(basis, ep.query_x), ep.query_y, p, sigma2)
            values.append(float(nll))
    return sum(values) / max(1, len(values))
