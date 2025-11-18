from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import fnmatch
import yaml
from .utils import BITS_CHOICES

# constaints engine for bitwidth vector
# takes raw per-unit bit assignment and makes them match the YAML file
# main loop: takes raw per-unit bits, makes them legal + structured

# should pull from the YAML structure
@dataclass
class GroupSpec:
    units: List[str] # explicit layer names: layers.0.attn.q_proj
    groups: List[List[str]] # list of patterns to group together # e.g. [["layers.*.attn.q_proj", "layers.*.attn.k_proj"]]
    exclude: List[str] # patterns to exclude from quantization
    choices: List[int] # the allowed bitWidths

"""
Problem: given something like 32 transformer layers and each layer has 6 weight matrices,
you are working with 200+ matrices. Each individual vector assigned with the accepted bit ranges creates
a ton of need to high amount of search. 

Solution: Q,K,V projections should be the same gene for every layer. 



unit_names: list of all quantizible unit names in order
self.spec: GroupSpec loaded from YAML
"""
class GroupTies:
    def __init__(self, unit_names: List[str], spec_path: str):
        with open(spec_path, 'r') as f:
            raw = yaml.safe_load(f)
        self.spec = GroupSpec(
            units=raw.get("units", []),
            groups=raw.get("groups", []),
            exclude=raw.get("exclude", []),
            choices=raw.get("choices", list(BITS_CHOICES))
        )

        self.unit_names = unit_names
        self.L = len(unit_names)
        self.excluded_mask = [self._match_any(n, self.spec.exclude) for n in unit_names]

        self.tie_groups: List[List[int]] = []

        # create all the lists of indices where layers are tied together
        for grp in self.spec.groups:
            indices: List[int] = []
            for pattern in grp:
                for i, name in enumerate(unit_names):
                    if fnmatch.fnmatch(name, pattern):
                        if i not in indices:
                            indices.append(i)
            indices = sorted(set(indices))
            if indices:
                self.tie_groups.append(indices)
        
        self.rep_for: Dict[int, int] = {}
        for indices in self.tie_groups:
            rep = indices[0]
            for i in indices:
                self.rep_for[i] = rep
    
    def _match_any(self, name: str, patterns: List[str]) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    # applies the group constraints 
    # for a q,k,v, you ensure they are the same.
    def repair(self, bits: Tuple[int, ...]) -> Tuple[int, ...]:
        choices = self.spec.choices 
        out = list(bits)

        for i, b in enumerate(out):
            if b not in choices:
                out[i] = min(choices, key=lambda c: abs(c - b))
        
        for indices in self.tie_groups:
            rep_b = out[indices[0]]
            for i in indices[1:]:
                out[i] = rep_b 
        return tuple(out)
