from __future__ import annotations
import hashlib
import random
from typing import Tuple

# Valid bit-width choices (exported constant)
BITS_CHOICES: Tuple[int, ...] = (4, 8, 16)


def sha1_hash(data: bytes) -> str:
    """Compute SHA-1 hash of the given data."""
    return hashlib.sha1(data).hexdigest()


def create_individual_id(bits: Tuple[int, ...], group_version: str = "v1") -> str:
    h = hashlib.sha1()
    h.update(bytes(bits))
    h.update(group_version.encode())
    return h.hexdigest()[:16]


def set_seed(seed: int):
    random.seed(seed)
