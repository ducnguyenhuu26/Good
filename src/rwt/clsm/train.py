"""Reference-only initialization; not a complete CLSM generalized-EM trainer."""
from __future__ import annotations

import torch
from torch import Tensor


def train_reference(model, obs: Tensor, action: Tensor, next_obs: Tensor,
                    *, epochs: int = 10, lr: float = 3e-4, seed: int = 41) -> list[float]:
    if epochs < 1 or lr <= 0:
        raise ValueError("positive training budget required")
    if obs.ndim != 2 or action.ndim != 2 or next_obs.shape != obs.shape or len(obs) != len(action):
        raise ValueError("reference transition batch mismatch")
    obs, action, next_obs = (x.to(torch.float64) for x in (obs, action, next_obs))
    # The caller must supply a separately collected, known-nominal reference
    # dataset. No evaluator intervention label enters the model objective.
    model.fit_normalization(obs, action, next_obs)
    generator = torch.Generator(device=obs.device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    code = torch.zeros(len(obs), dtype=torch.long, device=obs.device)
    theta = torch.zeros((len(obs),3), dtype=torch.float64, device=obs.device)
    curve = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(obs), generator=generator, device=obs.device)
        for indices in order.split(min(256, len(obs))):
            predicted = model(obs[indices], action[indices], code[indices], theta[indices])[0]
            loss = ((predicted - next_obs[indices]) / model.delta_scale).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite reference loss")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite reference gradient")
            optimizer.step()
        with torch.no_grad():
            predicted = model(obs, action, code, theta)[0]
            curve.append(float(((predicted - next_obs) / model.delta_scale).square().mean()))
    return curve
