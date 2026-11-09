"""
Loads an experiment configuration from a YAML file.

Paths in the config (parameter counts, unit specs, the latency LUT, the
output directory) are resolved relative to the directory containing the
config file, so a run behaves the same no matter which directory you
invoke it from.
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import yaml
from dataclasses import dataclass


class LatencyConfig(Enum):
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
    seed: int


# single evaluation loop
@dataclass
class EvalConfig:
    # how many validation tokens to use when evaluating the model
    val_tokens_schedule: tuple[int, int]  # [512, 2048] -> early generations evaluate 512 and 2048 for a more refined search
    # can speed up search using a LUT file
    measure_latency: LatencyConfig
    save_checkpoints: str  # best, all, none
    # when True, skip loading the model and evaluate individuals with a
    # deterministic synthetic loss instead. Only for CPU smoke tests of the
    # search machinery; the numbers it produces are not model results.
    mock_backend: bool = False


@dataclass
class ModelConfig:
    name: str
    param_counts_file: str | None  # path to csv or json mapping layers to parameter count
    default_fp_bits: int  # whatever the model says


@dataclass
class UnitsConfig:
    spec_file: str
    layer_spec_file: str


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


def _resolve(base: Path, path: str | None) -> str | None:
    """Resolve a config-relative path against the config file's directory."""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(base / p)


def load_config(path: str) -> GlobalConfig:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    base = Path(path).resolve().parent

    run_config = RunConfig(**raw["run"])
    eval_raw: Dict[str, Any] = dict(raw["evaluation"])
    vts = eval_raw.get("val_tokens_schedule")
    if isinstance(vts, list):
        eval_raw["val_tokens_schedule"] = tuple(vts)
    ml = eval_raw.get("measure_latency")
    if isinstance(ml, str):
        eval_raw["measure_latency"] = LatencyConfig(ml)
    eval_raw.setdefault("mock_backend", False)
    eval_config = EvalConfig(**eval_raw)

    model_raw: Dict[str, Any] = dict(raw["model"])
    model_raw["param_counts_file"] = _resolve(base, model_raw.get("param_counts_file"))
    model_config = ModelConfig(**model_raw)

    unit_raw: Dict[str, Any] = dict(raw["units"])
    unit_raw["spec_file"] = _resolve(base, unit_raw["spec_file"])
    unit_raw["layer_spec_file"] = _resolve(base, unit_raw["layer_spec_file"])
    unit_config = UnitsConfig(**unit_raw)

    hw_config = HardwareConfig(lut_file=_resolve(base, raw["hardware"]["lut_file"]))
    io_config = IOConfig(out_dir=_resolve(base, raw["io"]["out_dir"]))

    return GlobalConfig(
        run=run_config,
        evaluation=eval_config,
        model=model_config,
        units=unit_config,
        hardware=hw_config,
        io=io_config,
    )
