"""Build all three linked CLSM layers from one resolved configuration."""
from __future__ import annotations

from dataclasses import dataclass

from .dynamics import ExecutableLaw
from .evolution import LawEvolution
from .forecast import AgeLawForecaster
from .inference_dp import AgeLawFilter
from .memory import PopulationHierarchy, WorldTheoryMemory
from .schemas import EnvSchema


@dataclass
class CLSM:
    law: ExecutableLaw
    evolution: LawEvolution
    hierarchy: PopulationHierarchy
    memory: WorldTheoryMemory
    filter: AgeLawFilter
    forecaster: AgeLawForecaster


def build_clsm(schema: EnvSchema, config: dict) -> CLSM:
    network, emission = (config[k] for k in ("network", "emission"))
    evolution_config, memory_config = config["evolution"], config["memory"]
    filter_config, forecast_config = config["filter"], config["forecast"]
    law = ExecutableLaw(schema, network["hidden_dim"], emission["variance_floor"],
                        emission["variance_ceiling"])
    evolution = LawEvolution(evolution_config["initial_parameter_reset_prob"],
                             evolution_config["q_min"],
                             evolution_config["duration_scale_init"],
                             evolution_config["duration_shape_init"])
    hierarchy = PopulationHierarchy(memory_config["population_variance_floor"],
                                    memory_config["within_visit_variance_floor"])
    memory = hierarchy.new_world()
    filt = AgeLawFilter(law, evolution, memory, filter_config["max_world_length"],
                        config["initial_law_information"])
    forecaster = AgeLawForecaster(law, evolution, memory, forecast_config["samples"])
    return CLSM(law, evolution, hierarchy, memory, filt, forecaster)
