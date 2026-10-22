"""Evaluator-only world schedules; never included in learner datasets."""
from __future__ import annotations

import numpy as np

from experiments.rwt.clsm.physics_wrappers import HiddenLaw


COMP_TRAIN_CODES = (0, 1, 2, 4)
ALL_CODES = tuple(range(8))


def allowed_codes(track: str, split: str) -> tuple[int, ...]:
    if track == "comp" and split != "test":
        return COMP_TRAIN_CODES
    if track == "mem" and split != "test":
        return (0, 1, 2, 3, 4, 5)
    return ALL_CODES


def make_schedule(track: str, split: str, length: int, seed: int,
                  condition: str = "in_family") -> list[HiddenLaw]:
    if length < 1 or condition not in ("in_family", "misspecified", "stationary",
                                      "mem_independent", "mem_correlated"):
        raise ValueError("invalid schedule request")
    named_mem = condition in ("mem_independent", "mem_correlated")
    if named_mem and track != "mem":
        raise ValueError("named recurrence conditions are MEM-only benchmark amendments")
    rng = np.random.default_rng(seed)
    strength_rng = np.random.default_rng(seed + 7000000) if named_mem else rng
    allowed = allowed_codes(track, split)
    # MEM-R is a separate benchmark condition. Persistent, evaluator-only
    # physical prototypes are sampled once per world, then revisits draw a
    # nearby strength. The original independent-entry process stays intact.
    bounds_low = np.array([.50, .50, 1.30])
    bounds_high = np.array([.85, .85, 2.0])
    prototypes = ({code: strength_rng.uniform(bounds_low, bounds_high) for code in allowed if code != 0}
                  if condition == "mem_correlated" else {})

    def entry_strength(target: int) -> np.ndarray:
        if condition not in ("mem_independent", "mem_correlated"):
            return np.array([rng.uniform(.50,.85), rng.uniform(.50,.85), rng.uniform(1.30,2.0)])
        if target == 0:
            return np.ones(3)
        if condition == "mem_independent":
            return strength_rng.uniform(bounds_low, bounds_high)
        return np.clip(prototypes[target] * np.exp(strength_rng.normal(0, .04, 3)),
                       bounds_low, bounds_high)

    code = 0
    age = 0
    strength = np.ones(3)
    schedule = []
    for t in range(length):
        # Physical values remain evaluator-only. Same-code reset changes them
        # without resetting structural age.
        if t and condition != "stationary":
            scale, shape = (20., 1.4) if condition == "in_family" else (23., .8)
            hazard = -np.expm1(-(((age + 1) / scale) ** shape - (age / scale) ** shape))
            if rng.random() < hazard and len(allowed) > 1:
                choices = np.array([x for x in allowed if x != code], dtype=int)
                distance = np.array([(x ^ code).bit_count() for x in choices])
                preference = np.exp(-1.3 * distance)
                if condition == "misspecified":
                    preference *= 1 + .15 * (choices % 2)
                code = int(rng.choice(choices, p=preference / preference.sum()))
                age = 0
                reset = False
                strength = entry_strength(code)
            else:
                age += 1
                if rng.random() < .05 and not (named_mem and code == 0):
                    strength = entry_strength(code)
                    reset = True
                else:
                    if not named_mem or code != 0:
                        strength[:2] = np.clip(strength[:2] * np.exp(strength_rng.normal(0,.01,2)),.50,.85)
                        strength[2] = np.clip(strength[2] * np.exp(strength_rng.normal(0,.01)),1.30,2.0)
                    reset = False
        else:
            reset = False
        schedule.append(HiddenLaw(code, float(strength[0]), float(strength[1]),
                                  float(strength[2]), reset))
    return schedule
