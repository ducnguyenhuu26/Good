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
    if length < 1 or condition not in ("in_family", "misspecified", "stationary"):
        raise ValueError("invalid schedule request")
    rng = np.random.default_rng(seed)
    allowed = allowed_codes(track, split)
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
                strength = np.array([rng.uniform(.50,.85), rng.uniform(.50,.85), rng.uniform(1.30,2.0)])
            else:
                age += 1
                if rng.random() < .05:
                    strength = np.array([rng.uniform(.50,.85), rng.uniform(.50,.85), rng.uniform(1.30,2.0)])
                    reset = True
                else:
                    strength[:2] = np.clip(strength[:2] * np.exp(rng.normal(0,.01,2)),.50,.85)
                    strength[2] = np.clip(strength[2] * np.exp(rng.normal(0,.01)),1.30,2.0)
                    reset = False
        else:
            reset = False
        schedule.append(HiddenLaw(code, float(strength[0]), float(strength[1]),
                                  float(strength[2]), reset))
    return schedule
