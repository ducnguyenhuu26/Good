import itertools
import math

import numpy as np
import torch

from rwt.clsm.evolution import LawEvolution
from rwt.clsm.inference_dp import (AgeLawFilter, run_length_event_forward_backward,
                                   run_length_forward_backward)


def _enumerate_paths(log_emission, initial, hazard, transition):
    horizon = len(log_emission)
    rows = []

    def visit(path, ages, log_weight):
        t = len(path) - 1
        code = path[-1]
        log_weight += float(log_emission[t, code])
        if t == horizon - 1:
            rows.append((path[:], ages[:], log_weight))
            return
        age = ages[-1]
        h = float(hazard[code, age])
        if h < 1:
            visit(path + [code], ages + [age + 1], log_weight + math.log1p(-h))
        if h > 0:
            for target in range(8):
                p = float(transition[code, target])
                if target != code and p > 0:
                    visit(path + [target], ages + [0], log_weight + math.log(h) + math.log(p))

    for code in range(8):
        for age in range(initial.shape[1]):
            p = float(initial[code, age])
            if p > 0:
                visit([code], [age], math.log(p))
    logz = torch.logsumexp(torch.tensor([row[2] for row in rows], dtype=torch.float64), 0)
    weights = torch.tensor([math.exp(row[2] - float(logz)) for row in rows], dtype=torch.float64)
    smoothed = torch.zeros((horizon, 8, horizon), dtype=torch.float64)
    pairwise = torch.zeros((horizon - 1, 8, horizon, 8, 2), dtype=torch.float64)
    for weight, (codes, ages, _) in zip(weights, rows):
        for t, (code, age) in enumerate(zip(codes, ages)):
            smoothed[t, code, age] += weight
        for t in range(horizon - 1):
            event = 0 if codes[t] == codes[t + 1] else 1
            pairwise[t, codes[t], ages[t], codes[t + 1], event] += weight
    return logz, smoothed, pairwise


def test_exact_age_dp_matches_path_enumeration_with_code_emissions():
    horizon = 5
    initial = torch.zeros((8, horizon), dtype=torch.float64)
    initial[0, 0] = 1.
    hazard = torch.full((8, horizon), .23, dtype=torch.float64)
    transition = torch.ones((8, 8), dtype=torch.float64)
    transition.fill_diagonal_(0.)
    transition /= transition.sum(1, keepdim=True)
    emissions = torch.linspace(-.4, .5, horizon * 8, dtype=torch.float64).reshape(horizon, 8)
    actual = run_length_forward_backward(emissions, initial, hazard, transition)
    expected = _enumerate_paths(emissions, initial, hazard, transition)
    assert torch.allclose(actual.log_evidence, expected[0], atol=1e-12, rtol=0)
    assert torch.allclose(actual.smoothed, expected[1], atol=1e-12, rtol=0)
    assert torch.allclose(actual.pairwise_events, expected[2], atol=1e-12, rtol=0)
    assert torch.allclose(actual.smoothed.sum((1, 2)), torch.ones(horizon, dtype=torch.float64), atol=1e-12)


def test_500_step_exact_age_dp_matches_independent_probability_reference():
    rng = np.random.default_rng(3301)
    horizon = 500
    log_emissions = torch.tensor(rng.normal(0., .2, (horizon, 8)), dtype=torch.float64)
    hazard = torch.tensor(rng.uniform(.001, .2, (8, horizon)), dtype=torch.float64)
    transition_np = rng.uniform(.1, 1., (8, 8))
    np.fill_diagonal(transition_np, 0.)
    transition_np /= transition_np.sum(1, keepdims=True)
    transition = torch.tensor(transition_np, dtype=torch.float64)
    initial = torch.zeros((8, horizon), dtype=torch.float64)
    initial[0, 0] = 1.

    actual = run_length_forward_backward(log_emissions, initial, hazard, transition)
    reference = np.zeros((8, horizon), dtype=np.float64)
    reference[0, 0] = 1.
    filtered = np.zeros((horizon, 8, horizon), dtype=np.float64)
    total_log_evidence = 0.
    for t in range(horizon):
        likelihood = np.exp(log_emissions[t].numpy() - log_emissions[t].max().item())
        reference *= likelihood[:, None]
        scale = reference.sum()
        reference /= scale
        total_log_evidence += log_emissions[t].max().item() + math.log(scale)
        filtered[t] = reference
        if t == horizon - 1:
            break
        next_reference = np.zeros_like(reference)
        for code in range(8):
            for age in range(t + 1):
                mass = reference[code, age]
                h = float(hazard[code, age])
                next_reference[code, age + 1] += mass * (1. - h)
                next_reference[:, 0] += mass * h * transition_np[code]
        reference = next_reference

    assert abs(float(actual.log_evidence) - total_log_evidence) < 1e-10
    assert np.max(np.abs(actual.filtered.numpy() - filtered)) < 1e-10
    assert torch.max(torch.abs(actual.smoothed.sum((1, 2)) - 1.)) < 1e-12
    assert torch.max(torch.abs(actual.pairwise_events.sum((1, 2, 3, 4)) - 1.)) < 1e-10


def test_event_conditioned_smoother_matches_exhaustive_reset_and_switch_paths():
    horizon = 4
    initial = torch.zeros((8, horizon), dtype=torch.float64)
    initial[0, 0] = 1.
    hazard = torch.full((8, horizon), .2, dtype=torch.float64)
    transition = torch.ones((8, 8), dtype=torch.float64)
    transition.fill_diagonal_(0.)
    transition /= transition.sum(1, keepdim=True)
    rho = torch.full((8,), .3, dtype=torch.float64)
    rho[0] = .9  # must be overridden to zero by the implementation
    emission = torch.linspace(-.25, .35, horizon * 8 * 3, dtype=torch.float64).reshape(horizon, 8, 3)
    actual = run_length_event_forward_backward(emission, initial, hazard, transition, rho)

    rows = []

    def visit(codes, ages, events, score):
        t = len(codes) - 1
        if t == 0:
            score += float(emission[t, codes[-1], 0])
        if t == horizon - 1:
            rows.append((codes, ages, events, score))
            return
        code, age = codes[-1], ages[-1]
        h = float(hazard[code, age])
        reset = 0. if code == 0 else float(rho[code])
        if h < 1:
            stay = math.log1p(-h)
            for event, event_prob in ((0, 1 - reset), (1, reset)):
                if event_prob > 0:
                    visit(codes + [code], ages + [age + 1], events + [event],
                          score + stay + math.log(event_prob) + float(emission[t + 1, code, event]))
        if h > 0:
            for target in range(8):
                p = float(transition[code, target])
                if target != code and p > 0:
                    visit(codes + [target], ages + [0], events + [2],
                          score + math.log(h) + math.log(p) + float(emission[t + 1, target, 2]))

    visit([0], [0], [], 0.)
    logz = torch.logsumexp(torch.tensor([row[3] for row in rows], dtype=torch.float64), 0)
    weights = [math.exp(row[3] - float(logz)) for row in rows]
    smoothed = torch.zeros_like(actual.smoothed)
    pairwise = torch.zeros_like(actual.pairwise_events)
    for weight, (codes, ages, events, _) in zip(weights, rows):
        for t, (code, age) in enumerate(zip(codes, ages)):
            smoothed[t, code, age] += weight
        for t, event in enumerate(events):
            target = codes[t + 1]
            pairwise[t, codes[t], ages[t], target, event] += weight
    assert abs(float(actual.log_evidence - logz)) < 1e-12
    assert torch.allclose(actual.smoothed, smoothed, atol=1e-12, rtol=0)
    if not torch.allclose(actual.pairwise_events, pairwise, atol=1e-12, rtol=0):
        diff = (actual.pairwise_events - pairwise).abs()
        idx = tuple(int(i) for i in torch.unravel_index(diff.argmax(), diff.shape))
        raise AssertionError((float(diff.max()), idx, float(actual.pairwise_events[idx]), float(pairwise[idx])))
    assert float(actual.event_counts["stay_reset"]) < 1.


class _Memory:
    def entry_prior(self, code):
        mean = torch.zeros(3, dtype=torch.float64)
        cov = torch.diag(torch.tensor([float((code >> j) & 1) for j in range(3)], dtype=torch.float64))
        return mean, cov


class _LinearLaw:
    def __call__(self, obs, action, code, xi):
        bits = ((code[:, None] >> torch.arange(3, device=code.device)) & 1).to(obs.dtype)
        mean = obs + (xi * bits).sum(-1, keepdim=True) * .1
        return mean, torch.full_like(mean, .2)


def test_age_filter_keeps_exact_age_support_and_disables_nominal_reset():
    evolution = LawEvolution()
    filt = AgeLawFilter(_LinearLaw(), evolution, _Memory(), max_world_length=10,
                        initial_law_information="known_nominal")
    obs = torch.zeros(1, dtype=torch.float64)
    action = torch.zeros(1, dtype=torch.float64)
    filt.observe(obs, action, obs)
    branch = filt.evolve()
    assert abs(branch["mass"] - 1.) < 1e-12
    assert all(not (row.code == 0 and row.event == "stay_reset")
               for row in filt.pending)
    filt.observe(obs, action, obs)
    assert abs(float(filt.probabilities.sum()) - 1.) < 1e-12
    assert torch.isfinite(filt.log_code_age[:, :2]).any()
    assert torch.isneginf(filt.log_code_age[:, 3:]).all()
    cloned = filt.clone_causal()
    assert torch.equal(cloned.log_code_age, filt.log_code_age)
    assert cloned.pending is None

