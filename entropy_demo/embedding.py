"""Deterministic hashing bag-of-words embedding to R^d."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "may",
    "might",
    "not",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    # Structural markers commonly present in this demo's memory strings.
    "fact",
    "hint",
    "true",
    "false",
}


def tokenize(
    text: str,
    *,
    min_token_len: int = 1,
    use_english_stopwords: bool = False,
) -> list[str]:
    """Tokenize into lowercase alphanumerics.

    Optional filters are provided for engineering experiments. Defaults match
    the original minimal spec (no filtering).
    """

    toks = _TOKEN_RE.findall(text.lower())
    if min_token_len <= 1 and not use_english_stopwords:
        return toks

    out: list[str] = []
    for t in toks:
        if len(t) < min_token_len:
            continue
        if use_english_stopwords and t in _EN_STOPWORDS:
            continue
        out.append(t)
    return out


def _token_to_index_and_sign(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:8], "little", signed=False) % dim
    sign = 1.0 if (digest[8] & 1) == 1 else -1.0
    return idx, sign


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def embed_text(
    text: str,
    *,
    dim: int = 256,
    token_min_len: int = 1,
    use_english_stopwords: bool = False,
) -> np.ndarray:
    """Embed text into a deterministic L2-normalized vector."""

    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text, min_token_len=token_min_len, use_english_stopwords=use_english_stopwords):
        idx, sign = _token_to_index_and_sign(tok, dim)
        vec[idx] += sign
    return l2_normalize(vec)


def embed_average(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Average multiple vectors and L2-normalize the result."""

    vecs = list(vectors)
    if not vecs:
        raise ValueError("embed_average requires at least one vector")
    out = np.zeros_like(vecs[0], dtype=np.float32)
    for v in vecs:
        out += v.astype(np.float32, copy=False)
    return l2_normalize(out)


def cos(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity (assumes vectors are L2-normalized)."""

    if u.size == 0 or v.size == 0:
        return 0.0
    return float(np.dot(u, v))
