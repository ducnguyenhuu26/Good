"""Build all three linked CLSM layers from one resolved configuration."""
from __future__ import annotations

from dataclasses import dataclass

from .belief import LawFilter
from .dynamics import ExecutableLaw
from .evolution import LawEvolution
from .forecast import LawForecaster
from .memory import TheoryMemory
from .schemas import EnvSchema


@dataclass
class CLSM:
    law: ExecutableLaw
    evolution: LawEvolution
    memory: TheoryMemory
    filter: LawFilter
    forecaster: LawForecaster


def build_clsm(schema: EnvSchema, config: dict) -> CLSM:
    network, mechanics, emission = (config[k] for k in ("network", "mechanisms", "emission"))
    evolution_config, memory_config = config["evolution"], config["memory"]
    filter_config, forecast_config = config["filter"], config["forecast"]
    law = ExecutableLaw(schema, network["hidden_dim"], emission["variance_floor"],
                        emission["variance_ceiling"], mechanics["delta_min"])
    evolution = LawEvolution(evolution_config["initial_parameter_reset_prob"],
                             evolution_config["q_min"])
    memory = TheoryMemory(evolution_config["active_theta_prior_mean"],
                          evolution_config["active_theta_prior_std"],
                          scratch_weight=memory_config["scratch_prior_weight"])
    filt = LawFilter(law, evolution, memory, filter_config["components"],
                     filter_config["max_dropped_mass"])
    forecaster = LawForecaster(law, evolution, memory, forecast_config["samples"])
    return CLSM(law, evolution, memory, filt, forecaster)
