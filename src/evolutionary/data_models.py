from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import json

# just type aliases
Bit = int  # bit width value
Gene = Bit  # genes are basically bits


@dataclass(frozen=True)
class Individual:
    bits: Tuple[Gene, ...]  # a genome - like [2, 4, 2, 8] where each layer has a bit width
    seed_tag: str  # what generated this individual - e.g. "randmix", "uniform8", "xover_mut"
    id: str

@dataclass
class Fitness:
    val_loss: float # validation loss
    mem_bytes: int
    latency_ms: float # latency in milliseconds

@dataclass
class EvaluatedIndividual:
    ind: Individual
    fit: Fitness
    aux: Dict[str, Any] # extra data from evaluation 

@dataclass
class OpSpec:
    op_type: str 
    m: int 
    n: int 
    k: int

# maps the model to the quantization config
# represents the ENTIRE model, all of the layer
@dataclass
class ParamLayout:
    units: List[str]           # unit names in order
    param_counts: List[int]    # same length as units
    excluded_mask: List[bool]  # True if unit excluded (fixed to fp bits). Which layers should be quantized
    unit_ops: Dict[str, List[OpSpec]]

def write_metadata(path:str, 
                   bits: Tuple[int, ...], 
                   units: list[str],
                   scheme: str,
                   val_loss: float,
                   mem_bytes: int, 
                   latency: int,
                   model_name: str="Llama_8b", 
                   ) -> None:
    meta = {
        "model_name": model_name,
        "bits": list(bits),
        "units": units,
        "scheme": scheme,
        "val_loss": val_loss,
        "mem_bytes": mem_bytes,
        "latency_ms": latency,
    }

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)






