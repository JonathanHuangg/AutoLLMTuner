"""
Loads the parameters from all of the yaml files
Currently does not work
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Any, Dict
from enum import Enum


class latency_config(Enum):
    REAL = "real"
    LOOKUP = "lookup"
    NONE = "none"

# main configuration for the evolutionary algorithm
@dataclass
class RunConfig:
    population_size: int 
    generations: int 
    mutation_rate: float 
    crossover_probability: float
    elitism_frac: float 
    threads: int 
    seed: int 

# single evaluation loop
@dataclass
class EvalConfig:
    # how many validation tokens to use when evaluating the model
    val_tokens_schedule: tuple[int, int] # [512, 2048] -> early generations evaluate 512 and 2048 for a more refined search
    # can speed up search using a LUT file
    measure_latency: latency_config
    save_checkpoints: str # best, all, none 

@dataclass
class ModelConfig:
    name: str 
    param_counts_file: str | None # path to csv or json mapping layers to parameter count
    default_fp_bits: int # whatever the model says

@dataclass
class UnitsConfig:
    spec_file: str

# hardware cost model -> converting bit allocations to latency/memory estimates
@dataclass
class HardwareConfig:
    lut_file: str

# specifies where the outputs all go
@dataclass
class IOConfig:
    out_dir: str

@dataclass 
class GlobalConfig:
    run: RunConfig
    evaluation: EvalConfig
    model: ModelConfig
    units: UnitsConfig 
    hardware: HardwareConfig
    io: IOConfig

def load_config(path: str) -> GlobalConfig:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    
    run_config = RunConfig(**raw["run"])
    eval_config = EvalConfig(**raw["evaluation"])
    model_config = ModelConfig(**raw["model"])
    unit_config = UnitsConfig(**raw["units"])
    hw_config = HardwareConfig(**raw["hardware"])
    io_config = IOConfig(**raw["io"])

    return GlobalConfig(run=run_config, 
                        evaluation=eval_config, 
                        model=model_config, 
                        units=unit_config, 
                        hardware=hw_config, 
                        io=io_config)


