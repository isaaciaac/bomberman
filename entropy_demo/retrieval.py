"""S1 retrieval: Q -> finite subset of the memory store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embedding import cos, embed_text
from .types import MemoryAtom


@dataclass(frozen=True)
class RetrievalResult:
    z_q: np.ndarray
    M0: list[MemoryAtom]


def s1_retrieve(
    q: str,
    memory_store: Sequence[MemoryAtom],
    *,
    k: int,
    dim: int = 256,
) -> RetrievalResult:
    """Retrieve top-K by cos(z_q, z_m) * max(0, s_i)."""

    z_q = embed_text(q, dim=dim)

    scored: list[tuple[float, MemoryAtom]] = []
    for m in memory_store:
        score = cos(z_q, m.z_i) * max(0.0, float(m.s_i))
        if score > 0.0:
            scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return RetrievalResult(z_q=z_q, M0=[m for _, m in scored[:k]])
