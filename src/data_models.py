from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

# just type aliases
Bit = int # bit width value
Gene = Bit # genes are basically bits

class SeedTag(str, Enum):
    UNIFORM8 = "uniform8"
    MUTATION = "mut"
    CROSSOVER = "xover"
    ELITE = "elite"
    RANDOM = "rand"

@dataclass(frozen=True)
class Individual:
    bits: Tuple[Gene, ...] # a genome - like [2, 4, 2, 8] where each layer has a bit width
    seed_tag: SeedTag # what generated this individual - e.g. "random", "mutated", "crossover"
    id: str 

@dataclass
class Fitness:
    val_loss: float # validation loss
    mem_bytes: int
    latency: float # latency in milliseconds

@dataclass
class EvaluatedIndividual:
    ind: Individual
    fit: Fitness
    aux: Dict[str, Any] # extra data from evaluation 

@dataclass
class ParamLayout:
    units: List[str]           # unit names in order
    param_counts: List[int]    # same length as units
    excluded_mask: List[bool]  # True if unit excluded (fixed to fp bits)


    




