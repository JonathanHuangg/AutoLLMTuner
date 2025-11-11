from __future__ import annotations
import hashlib
import random 
from typing import Iterable, Tuple 

# can later call random.choice(BITS_CHOICES) to get a random bit width
BITS_CHOICES = Tuple[int, ...] = (2, 4, 8, 16, 32)

# used for IDs of individuals
def sha1_hash(data: bytes):
    """Compute SHA-1 hash of the given data."""
    return hashlib.sha1(data).hexdigest()

def create_individual_id(bits: Tuple[int, ...], group_version: str = "v1") -> str:
    h = hashlib.sha1() 
    h.update(bytes(bits))
    h.update(group_version.encode())
    return h.hexdigest()[:16]

def set_seed(seed: int):
    random.seed(seed)

