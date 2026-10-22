"""Resolved, validated CLSM experiment configuration."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import yaml


DEFAULTS = {
    "network": {"hidden_dim": 256, "hidden_layers": 2, "activation": "silu", "reward_head": False},
    "mechanisms": {"n_motor_groups": 2, "damping_groups": 1, "delta_min": 0.12,
                   "sparse_weight": 0.001, "overlap_weight": 0.01,
                   "motor_change_direction": "weaken", "damping_change_direction": "increase"},
    "emission": {"diagonal_gaussian": True, "variance_floor": 1e-5,
                 "variance_ceiling": 1e2},
    "filter": {"components": 128, "dtype_probability": "float64",
               "max_dropped_mass": 0.001, "covariance_jitter_initial": 1e-9,
               "covariance_jitter_max": 1e-3, "support_preserve_by_code": True,
               "independent_forecast_rng": True},
    "evolution": {"duration": "discrete_weibull", "age_depends_on_popcount": True,
                  "sparse_edits": True, "same_structure_reset": True,
                  "initial_parameter_reset_prob": 0.05, "q_min": 1e-6,
                  "active_theta_prior_mean": -2.0, "active_theta_prior_std": 1.5},
    "memory": {"enabled": True, "scratch_prior_weight": 0.5,
               "reuse_prior_weight": 0.5, "max_theory_prototypes": 8,
               "visit_level_updates": True},
    "training": {"reference_epochs": 100, "outer_gem_iterations": 20,
                 "gradient_steps_per_iteration": 300, "batch_size": 256,
                 "lr_dynamics": 0.0003, "lr_evolution": 0.001,
                 "weight_decay": 0.0001,
                 "variance_penalty_weight": 0.01,
                 "validation_metric": "prequential_nll", "gradient_clip_norm": 5.0},
    "forecast": {"horizons": [1, 8, 24, 64], "samples": 256,
                 "signature_probes": 64},
}


def _merge_known(base: dict, overrides: dict, prefix: str = "") -> dict:
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"unknown CLSM hyperparameter {prefix + key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix + key} must be a mapping")
            _merge_known(base[key], value, prefix + key + ".")
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("method") != "clsm":
        raise ValueError("configuration must define method: clsm")
    # Base experimental fields are supplied per-track; all mathematical
    # hyperparameters are resolved here and persisted in run manifests.
    hyper = deepcopy(DEFAULTS)
    for section in DEFAULTS:
        if section in config:
            _merge_known(hyper[section], config.pop(section), section + ".")
    config.update(hyper)
    if config.get("track") not in ("comp", "evol", "mem"):
        raise ValueError("track must be comp, evol or mem")
    schedule_condition = config.get("schedule_condition", "in_family")
    if schedule_condition not in ("in_family", "misspecified", "stationary",
                                  "mem_independent", "mem_correlated"):
        raise ValueError("unknown schedule condition")
    if schedule_condition in ("mem_independent", "mem_correlated") and config["track"] != "mem":
        raise ValueError("named recurrence conditions are MEM-only benchmark amendments")
    if sorted(set(config.get("forecast_horizons", []))) != [1, 8, 24, 64] or config["forecast"]["horizons"] != [1, 8, 24, 64]:
        raise ValueError("forecast horizons must be [1,8,24,64]")
    counts = config.get("transitions", {})
    if any(not isinstance(counts.get(split), int) or counts[split] < 1 for split in ("train", "validation", "test")):
        raise ValueError("positive train/validation/test transition counts required")
    if not 0 < config.get("world_length", 0) <= 1000:
        raise ValueError("world_length must be in 1..1000")
    if config["network"] != {"hidden_dim": config["network"]["hidden_dim"],
                              "hidden_layers": 2, "activation": "silu", "reward_head": False}:
        raise ValueError("CLSM requires two SiLU layers and no reward head")
    if config["mechanisms"]["n_motor_groups"] != 2 or config["mechanisms"]["damping_groups"] != 1:
        raise ValueError("CLSM v1 requires exactly three mechanisms")
    if config["mechanisms"]["motor_change_direction"] != "weaken" or config["mechanisms"]["damping_change_direction"] != "increase":
        raise ValueError("intervention signs are locked")
    if config["filter"]["components"] < 16 or config["filter"]["dtype_probability"] != "float64":
        raise ValueError("filter requires float64 and support-preserving capacity")
    if config["memory"]["scratch_prior_weight"] + config["memory"]["reuse_prior_weight"] != 1:
        raise ValueError("scratch/reuse weights must sum to one")
    if not 1e-4 <= config["evolution"]["initial_parameter_reset_prob"] <= 0.25:
        raise ValueError("reset probability outside allowed range")
    if config["training"]["validation_metric"] != "prequential_nll":
        raise ValueError("checkpoint selection metric is locked")
    if not isinstance(config["network"]["hidden_dim"], int) or config["network"]["hidden_dim"] < 1:
        raise ValueError("network.hidden_dim must be positive")
    if not 0 < config["emission"]["variance_floor"] < config["emission"]["variance_ceiling"]:
        raise ValueError("emission variance range is invalid")
    if not 0 < config["filter"]["covariance_jitter_initial"] <= config["filter"]["covariance_jitter_max"]:
        raise ValueError("filter Cholesky jitter range is invalid")
    if not 0 <= config["filter"]["max_dropped_mass"] < 1:
        raise ValueError("filter mass-loss gate is invalid")
    for key in ("support_preserve_by_code", "independent_forecast_rng"):
        if config["filter"][key] is not True:
            raise ValueError(f"filter.{key} is required")
    if (config["evolution"]["duration"] != "discrete_weibull" or
            any(config["evolution"][key] is not True for key in
                ("age_depends_on_popcount", "sparse_edits", "same_structure_reset")) or
            config["evolution"]["q_min"] <= 0 or config["evolution"]["active_theta_prior_std"] <= 0):
        raise ValueError("required semi-Markov Gaussian evolution configuration invalid")
    if (config["memory"]["enabled"] is not True or config["memory"]["visit_level_updates"] is not True or
            config["memory"]["max_theory_prototypes"] != 8 or
            not 0 < config["memory"]["scratch_prior_weight"] < 1):
        raise ValueError("required hierarchical scratch/reuse memory configuration invalid")
    for key in ("reference_epochs", "outer_gem_iterations", "gradient_steps_per_iteration", "batch_size"):
        if not isinstance(config["training"][key], int) or config["training"][key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("lr_dynamics", "lr_evolution", "gradient_clip_norm"):
        if config["training"][key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("weight_decay", "variance_penalty_weight"):
        if config["training"][key] < 0:
            raise ValueError(f"training.{key} cannot be negative")
    if config["forecast"]["samples"] < 2 or config["forecast"]["signature_probes"] < 2:
        raise ValueError("forecast sample and probe counts must be at least two")
    mode = config.get("mode")
    if mode not in ("smoke", "full"):
        raise ValueError("experiment mode must be smoke or full")
    expected_collector = "smooth_random_smoke_v1" if mode == "smoke" else "nominal_controller_mixture_required"
    if config.get("collector") != expected_collector:
        raise ValueError("collector and experiment mode disagree")
    if mode == "full" and config.get("collector_min_nominal_steps", 100000) < 1:
        raise ValueError("full nominal collector budget must be positive")
    return config
