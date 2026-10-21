"""Public-posterior diagnostics emitted by training runs."""
from __future__ import annotations

import torch


def inference_diagnostic(components, time: int, dropped_mass: float) -> dict[str, float]:
    if not components:
        raise ValueError("inference diagnostics require posterior support")
    weights = torch.stack([component.log_weight for component in components]).to(torch.float64)
    weights = (weights - torch.logsumexp(weights, 0)).exp()
    row: dict[str, float] = {
        "time": float(time),
        "component_count": float(len(components)),
        "ess": float(weights.square().sum().reciprocal()),
        "dropped_mass": float(dropped_mass),
    }
    for code in range(8):
        row[f"code_weight_{code}"] = float(sum(
            weight for component, weight in zip(components, weights) if component.code == code))
    for hypothesis in (0, 1):
        row[f"model_weight_{hypothesis}"] = float(sum(
            weight for component, weight in zip(components, weights)
            if component.scratch_or_reuse == hypothesis))
    eigenvalues = []
    for component in components:
        active = [j for j in range(3) if (component.code >> j) & 1]
        if active:
            block = component.cov[active][:, active]
            eigenvalues.extend(torch.linalg.eigvalsh((block + block.T) / 2).tolist())
    row["active_cov_eig_min"] = float(min(eigenvalues)) if eigenvalues else 0.0
    row["active_cov_eig_max"] = float(max(eigenvalues)) if eigenvalues else 0.0
    return row


def evolution_diagnostic(components, evolution, time: int) -> dict[str, float]:
    if not components:
        raise ValueError("evolution diagnostics require posterior support")
    log_weights = torch.stack([component.log_weight for component in components]).to(torch.float64)
    weights = (log_weights - torch.logsumexp(log_weights, 0)).exp()
    codes = torch.tensor([component.code for component in components], dtype=torch.long)
    ages = torch.tensor([component.age for component in components], dtype=torch.long)
    hazards = evolution.hazard(codes, ages).detach()
    transition = evolution.structural_log_transition().exp().detach()
    reset_probability = evolution.reset_prob.detach()
    bits = torch.tensor([[(code >> j) & 1 for j in range(3)] for code in range(8)])
    distance = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    row = {
        "time": float(time),
        "expected_switch_probability": float(weights @ hazards),
        "expected_parameter_reset_probability": float(weights @ ((1 - hazards) * reset_probability)),
        "expected_drift_probability": float(weights @ ((1 - hazards) * (1 - reset_probability))),
        "weibull_shape_nu": float(evolution.nu.detach()),
        "reset_probability_given_stay": float(reset_probability),
    }
    for edits in (1, 2, 3):
        by_code = (transition * (distance == edits)).sum(-1)
        row[f"switch_mass_hamming_{edits}"] = float(weights @ (hazards * by_code[codes]))
    return row
