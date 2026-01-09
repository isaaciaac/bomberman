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
    token_min_len: int = 1,
    use_english_stopwords: bool = False,
    valence_mode: str = "clip_pos",
    score_threshold: float = 0.0,
) -> RetrievalResult:
    """Retrieve top-K by similarity, optionally influenced by directional weight.

    Scoring:
      base = cos(z_q, z_m)

    Valence modes:
      - clip_pos: score = base * max(0, s_i)
      - raw:      score = base * s_i
      - none:     score = base

    Items with score <= score_threshold are dropped before top-K selection.
    """

    z_q = embed_text(q, dim=dim, token_min_len=token_min_len, use_english_stopwords=use_english_stopwords)

    scored: list[tuple[float, MemoryAtom]] = []
    for m in memory_store:
        base = cos(z_q, m.z_i)
        if valence_mode == "none":
            score = base
        elif valence_mode == "raw":
            score = base * float(m.s_i)
        else:
            score = base * max(0.0, float(m.s_i))

        if score > score_threshold:
            scored.append((float(score), m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return RetrievalResult(z_q=z_q, M0=[m for _, m in scored[:k]])
