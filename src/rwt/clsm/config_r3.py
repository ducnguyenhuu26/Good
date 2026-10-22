"""Strict loader for the frozen CLSM-R3 final-attempt configuration."""
from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_CANDIDATE = "clsm_r3_intervention_resolved_nominal_anchored"


def validate_r3_config(config: dict) -> dict:
    if config.get("architecture_revision") != 3 or config.get("candidate_id") != REQUIRED_CANDIDATE:
        raise ValueError("R3 config loader rejects legacy or incompatible candidates")
    if config.get("environment") != "HalfCheetah-v5":
        raise ValueError("R3 final gate is registered only for HalfCheetah-v5")
    if config.get("mechanism_bits") != {"motor_group_1": 1, "motor_group_2": 2, "damping": 4}:
        raise ValueError("R3 mechanism bit semantics changed")
    if config.get("physical_bounds") != {"motor": [0.50, 0.85], "damping": [1.30, 2.00]}:
        raise ValueError("R3 physical strength bounds changed")
    decoder = config.get("decoder", {})
    if decoder.get("freeze_f0") is not True or decoder.get("nominal_anchor_subtraction") is not True:
        raise ValueError("R3 requires a frozen nominal anchor and exact residual subtraction")
    if decoder.get("residual_hidden_layers") != 2 or decoder.get("residual_activation") != "silu":
        raise ValueError("R3 residual architecture differs from the registered candidate")
    training = config.get("training", {})
    if training.get("train_seeds") != [4100, 4101, 4102] or training.get("optimizer") != "AdamW":
        raise ValueError("R3 training seeds or optimizer differ from registration")
    if (training.get("learning_rate") != 3e-4 or training.get("weight_decay") != 1e-4 or
            training.get("batch_size") != 256 or training.get("max_epochs") != 100 or
            training.get("early_stopping_patience") != 12):
        raise ValueError("R3 optimizer/training schedule differs from registration")
    evaluation = config.get("evaluation", {})
    if evaluation.get("diagnostic_codes") != [3, 5, 6, 7] or evaluation.get("bootstrap_unit") != "independent_world":
        raise ValueError("R3 diagnostic compositions or bootstrap unit changed")
    if evaluation.get("noninferiority_mse_fraction") != 0.05 or evaluation.get("noninferiority_signature_fraction") != 0.05:
        raise ValueError("R3 noninferiority margins changed")
    if config.get("provenance", {}).get("role") != "oracle_diagnostic" or not config["provenance"].get("oracle_labels_are_diagnostic_only"):
        raise ValueError("R3 oracle labels must remain diagnostic-only")
    return config


def load_r3_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_r3_config(yaml.safe_load(handle))
