"""Resolved, validated CLSM experiment configuration."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("method") != "clsm":
        raise ValueError("configuration must define method: clsm")
    if config.get("track") not in ("comp", "evol", "mem"):
        raise ValueError("track must be comp, evol or mem")
    if sorted(set(config.get("forecast_horizons", []))) != [1, 8, 24, 64]:
        raise ValueError("forecast horizons must be [1,8,24,64]")
    counts = config.get("transitions", {})
    if any(not isinstance(counts.get(split), int) or counts[split] < 1 for split in ("train", "validation", "test")):
        raise ValueError("positive train/validation/test transition counts required")
    if not 0 < config.get("world_length", 0) <= 1000:
        raise ValueError("world_length must be in 1..1000")
    return config
