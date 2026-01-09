"""External, deterministic verifier V(q, M) -> pass/fail.

The verifier is intentionally small and auditable. It is *not* the entropy gate.
It provides an additional, explicit acceptance signal for engineering validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import VerifierConfig
from .embedding import cos, embed_text, tokenize
from .entropy import chi
from .types import MemoryAtom, is_constraint_atom

_REQ_RE = re.compile(r"REQ\[([^\]]+)\]")

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
}


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]


def _is_non_generated(atom: MemoryAtom) -> bool:
    # Seed atoms usually do not carry rewrite_type.
    rt = atom.eta_i.get("rewrite_type", "")
    return not isinstance(rt, str) or rt == ""


def _atom_claim_key(atom: MemoryAtom) -> str | None:
    parsed = atom.eta_i.get("parsed") or {}
    key = parsed.get("key")
    if isinstance(key, str) and key:
        return key

    ck = atom.eta_i.get("claim_key", atom.eta_i.get("key"))
    if isinstance(ck, str) and ck:
        return ck
    return None


def _required_claim_keys_from_query(q: str) -> list[str]:
    """Extract explicit claim-key requirements from the query string.

    Supported syntax:
      ... REQ[key1, key2]
    """

    keys: list[str] = []
    for chunk in _REQ_RE.findall(q):
        for part in chunk.split(","):
            k = part.strip()
            if k:
                keys.append(k)
    # stable de-dup
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _content_tokens(text: str, *, cfg: VerifierConfig) -> set[str]:
    toks = tokenize(text)
    out: set[str] = set()
    for t in toks:
        if len(t) < cfg.evidence_token_min_len:
            continue
        if cfg.evidence_use_english_stopwords and t in _EN_STOPWORDS:
            continue
        out.add(t)
    return out


def verify_state(
    q: str,
    M: Sequence[MemoryAtom],
    *,
    cfg: VerifierConfig,
    z_q: np.ndarray | None = None,
) -> VerifierResult:
    """Deterministically verify whether (q, M) meets external acceptance rules."""

    if not cfg.enabled:
        return VerifierResult(passed=True, reasons=(), metrics={"enabled": False})

    reasons: list[str] = []

    non_generated = [m for m in M if _is_non_generated(m)]
    if len(non_generated) < cfg.min_non_generated_atoms:
        reasons.append(f"need >= {cfg.min_non_generated_atoms} non-generated atoms (got {len(non_generated)})")

    # Evidence token overlap requirement (computed over non-generated atoms).
    evidence_token_overlap_max = 0
    if cfg.min_evidence_token_overlap > 0:
        q_toks = _content_tokens(q, cfg=cfg)
        for m in non_generated:
            overlap = len(q_toks.intersection(_content_tokens(m.v_i, cfg=cfg)))
            evidence_token_overlap_max = max(evidence_token_overlap_max, int(overlap))
        if evidence_token_overlap_max < cfg.min_evidence_token_overlap:
            reasons.append(
                f"evidence_token_overlap_max<{cfg.min_evidence_token_overlap} (got {evidence_token_overlap_max})"
            )

    # Evidence similarity requirement (computed over non-generated atoms).
    evidence_sim_max = 0.0
    if cfg.min_evidence_sim > 0.0:
        if z_q is None:
            dim = int(M[0].z_i.shape[0]) if M else 256
            z_q = embed_text(q, dim=dim)
        sims: list[float] = []
        for m in non_generated:
            if m.z_i.shape != z_q.shape:
                continue
            sims.append(cos(z_q, m.z_i))
        if cfg.evidence_clip_cosine:
            sims = [max(0.0, s) for s in sims]
        evidence_sim_max = max(sims) if sims else 0.0
        if evidence_sim_max < cfg.min_evidence_sim:
            reasons.append(
                "evidence_sim_max<{:.3f} (got {:.3f})".format(cfg.min_evidence_sim, evidence_sim_max)
            )

    unreconciled: list[tuple[str, str]] = []
    if cfg.require_no_unreconciled_conflicts:
        items = [m for m in M if not is_constraint_atom(m)]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if chi(items[i], items[j], M=M) == 1:
                    unreconciled.append(tuple(sorted((items[i].id, items[j].id))))
        if unreconciled:
            reasons.append(f"unreconciled_conflicts={len(unreconciled)}")

    required_keys: list[str] = []
    if cfg.require_claim_keys_from_query:
        required_keys.extend(_required_claim_keys_from_query(q))
    required_keys.extend([str(k) for k in cfg.required_claim_keys if str(k)])
    missing_keys: list[str] = []
    if required_keys:
        present = {k for m in M if (k := _atom_claim_key(m)) is not None}
        for k in required_keys:
            if k not in present:
                missing_keys.append(k)
        if missing_keys:
            reasons.append(f"missing_required_claim_keys={missing_keys}")

    passed = len(reasons) == 0
    metrics: dict[str, Any] = {
        "enabled": True,
        "non_generated_atoms": int(len(non_generated)),
        "min_evidence_sim": float(cfg.min_evidence_sim),
        "evidence_sim_max": float(evidence_sim_max),
        "min_evidence_token_overlap": int(cfg.min_evidence_token_overlap),
        "evidence_token_overlap_max": int(evidence_token_overlap_max),
        "require_no_unreconciled_conflicts": bool(cfg.require_no_unreconciled_conflicts),
        "unreconciled_conflicts": int(len(unreconciled)),
        "required_claim_keys": list(required_keys),
        "missing_claim_keys": list(missing_keys),
    }
    if unreconciled:
        metrics["unreconciled_pairs"] = [list(p) for p in sorted(set(unreconciled))]
    return VerifierResult(passed=passed, reasons=tuple(reasons), metrics=metrics)
