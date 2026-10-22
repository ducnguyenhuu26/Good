"""Resolved and validated CLSM-R2 methodology configuration."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULTS = {
    "architecture_revision": 2,
    "network": {"hidden_dim": 256, "hidden_layers": 2, "activation": "silu", "reward_head": False},
    "mechanisms": {"n_motor_groups": 2, "damping_groups": 1},
    "emission": {"diagonal_gaussian": True, "variance_floor": 1e-5,
                 "variance_ceiling": 1e2, "nominal_reference_weight": 1.0},
    "partition": {"max_actuators": 8, "quadrature_nodes": 12,
                  "initialization_seeds": [41, 42, 43, 44, 45],
                  "update_period": 1, "train_world_bootstrap_replicates": 1000},
    "filter": {"max_world_length": 500, "dtype_probability": "float64",
               "age_representation": "exact_log_dp", "strength_update": "symmetric_cubature",
               "covariance_jitter_initial": 1e-10, "covariance_jitter_max": 1e-5,
               "mass_tolerance": 1e-10},
    "evolution": {"duration": "discrete_weibull", "age_depends_on_popcount": True,
                  "sparse_edits": True, "same_structure_reset": True,
                  "initial_parameter_reset_prob": 0.05, "q_min": 1e-6,
                  "duration_scale_init": 20.0, "duration_shape_init": 1.4},
    "memory": {"enabled": True, "population_variance_floor": 1e-4,
               "within_visit_variance_floor": 0.05, "visit_level_updates": True,
               "world_boundary_reset": True},
    "training": {"reference_epochs": 100, "outer_gem_iterations": 20,
                 "gradient_steps_per_iteration": 300, "batch_size": 256,
                 "lr_dynamics": 0.0003, "lr_evolution": 0.001,
                 "weight_decay": 0.0001, "variance_penalty_weight": 0.01,
                 "validation_metric": "prequential_nll", "gradient_clip_norm": 5.0,
                 "nominal_replay_ratio": 1.0},
    "forecast": {"horizons": [1, 8, 24, 64], "samples": 256,
                 "signature_probes": 64},
}


def _merge_known(base: dict, overrides: dict, prefix: str = "") -> None:
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"unknown CLSM-R2 hyperparameter {prefix + key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix + key} must be a mapping")
            _merge_known(base[key], value, prefix + key + ".")
        else:
            base[key] = value


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or raw.get("method") != "clsm":
        raise ValueError("configuration must define method: clsm")
    if raw.get("architecture_revision") != 2:
        raise ValueError("CLSM-R2 config must explicitly set architecture_revision: 2; legacy configs are unsupported")
    config = deepcopy(raw)
    hyper = deepcopy(DEFAULTS)
    for section in DEFAULTS:
        if section == "architecture_revision":
            continue
        if section in config:
            _merge_known(hyper[section], config.pop(section), section + ".")
    if config.pop("architecture_revision") != 2:
        raise ValueError("only CLSM-R2 architecture_revision 2 is supported")
    config.update(hyper)
    if config.get("track") not in ("comp", "evol", "mem"):
        raise ValueError("track must be comp, evol or mem")
    if config.get("initial_law_information") not in ("known_nominal", "unknown", "comp_design"):
        raise ValueError("initial_law_information must be declared as known_nominal, unknown, or comp_design")
    condition = config.get("schedule_condition", "in_family")
    if condition not in ("in_family", "misspecified", "stationary", "mem_independent", "mem_correlated"):
        raise ValueError("unknown schedule condition")
    if condition in ("mem_independent", "mem_correlated") and config["track"] != "mem":
        raise ValueError("named MEM recurrence conditions are MEM-only")
    if condition == "mem_correlated" and (config["evolution"]["duration_scale_init"] != 23.0 or
                                           config["evolution"]["duration_shape_init"] != 0.8):
        raise ValueError("mem_correlated must record its registered duration scale=23 and shape=0.8")
    if config.get("forecast_horizons") != [1, 8, 24, 64] or config["forecast"]["horizons"] != [1, 8, 24, 64]:
        raise ValueError("forecast horizons are locked to [1,8,24,64]")
    counts = config.get("transitions", {})
    if any(not isinstance(counts.get(split), int) or counts[split] < 1
           for split in ("train", "validation", "test")):
        raise ValueError("positive train/validation/test transition counts required")
    world_length = config.get("world_length")
    if not isinstance(world_length, int) or not 0 < world_length <= config["filter"]["max_world_length"] <= 500:
        raise ValueError("world length must fit the exact finite age table of at most 500 steps")
    if config["network"] != {"hidden_dim": config["network"]["hidden_dim"],
                              "hidden_layers": 2, "activation": "silu", "reward_head": False}:
        raise ValueError("CLSM-R2 retains the two-layer SiLU emission and no reward head")
    if config["mechanisms"] != {"n_motor_groups": 2, "damping_groups": 1}:
        raise ValueError("CLSM-R2 fixes the mechanism grammar to two motor groups and one damping mechanism")
    if config["emission"]["diagonal_gaussian"] is not True or not 0 < config["emission"]["variance_floor"] < config["emission"]["variance_ceiling"]:
        raise ValueError("invalid positive diagonal Gaussian emission bounds")
    if config["emission"]["nominal_reference_weight"] <= 0:
        raise ValueError("nominal reference replay weight must be positive")
    if config["partition"]["max_actuators"] > 8 or config["partition"]["max_actuators"] < 2:
        raise ValueError("finite partition search supports 2..8 actuators")
    if config["partition"]["quadrature_nodes"] != 12:
        raise ValueError("partition calibration uses the fixed 12-node Gauss-Hermite rule")
    if len(config["partition"]["initialization_seeds"]) > 5 or not config["partition"]["initialization_seeds"]:
        raise ValueError("predeclare one to five global partition initialization seeds")
    if config["filter"]["dtype_probability"] != "float64" or config["filter"]["age_representation"] != "exact_log_dp":
        raise ValueError("CLSM-R2 requires float64 exact log-space age recursion")
    if config["filter"]["strength_update"] != "symmetric_cubature":
        raise ValueError("CLSM-R2 requires positive-weight symmetric cubature")
    if config["filter"]["mass_tolerance"] > 1e-10 or config["filter"]["mass_tolerance"] <= 0:
        raise ValueError("code-age mass tolerance must not exceed 1e-10")
    if not 0 < config["filter"]["covariance_jitter_initial"] <= config["filter"]["covariance_jitter_max"]:
        raise ValueError("invalid cubature covariance jitter range")
    evolution = config["evolution"]
    if evolution["duration"] != "discrete_weibull" or any(evolution[k] is not True for k in
            ("age_depends_on_popcount", "sparse_edits", "same_structure_reset")):
        raise ValueError("required explicit-duration sparse-edit evolution is disabled")
    if not 0 < evolution["initial_parameter_reset_prob"] <= .25 or evolution["q_min"] <= 0:
        raise ValueError("invalid reset or drift variance floor")
    if evolution["duration_scale_init"] <= 0 or evolution["duration_shape_init"] <= 0:
        raise ValueError("Weibull scale and shape must be positive")
    memory = config["memory"]
    if memory["enabled"] is not True or memory["visit_level_updates"] is not True or memory["world_boundary_reset"] is not True:
        raise ValueError("R2 requires per-world visit-level hierarchical memory")
    if min(memory["population_variance_floor"], memory["within_visit_variance_floor"]) <= 0:
        raise ValueError("hierarchy variance floors must be positive")
    if config["training"]["validation_metric"] != "prequential_nll" or config["training"]["nominal_replay_ratio"] <= 0:
        raise ValueError("R2 checkpoint selection and nominal replay objective are locked")
    for key in ("reference_epochs", "outer_gem_iterations", "gradient_steps_per_iteration", "batch_size"):
        if not isinstance(config["training"][key], int) or config["training"][key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("lr_dynamics", "lr_evolution", "gradient_clip_norm"):
        if config["training"][key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if config["forecast"]["samples"] < 2 or config["forecast"]["signature_probes"] < 2:
        raise ValueError("forecast sample and probe counts must be at least two")
    if config.get("mode") not in ("smoke", "pilot", "full"):
        raise ValueError("mode must be smoke, pilot, or full")
    return config
