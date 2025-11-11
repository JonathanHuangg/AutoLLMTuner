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
    REAL = "real",
    LOOKUP = "lookup",
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

@dataclass
class HardwareConfig:
    lut_file: str

@dataclass
class IOConfig:
    out_dir: str
    
@dataclass 
class GlobalConfig:
    run: RunConfig
    eval: EvalConfig
    model: ModelConfig
    units: UnitsConfig 
    hardware: HardwareConfig
    io: IOConfig




