"""S2 rewrite proposals (REWRITE) that monotonically extend M and strictly decrease entropy."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import EntropyConfig, RewriteConfig, StabilityConfig
from .embedding import cos, embed_average, embed_text
from .entropy import EntropyBreakdown, EntropyWeights, HistoryStore, compute_entropy
from .types import MemoryAtom, is_constraint_atom


class AtomFactory:
    """Simple ID generator for new atoms."""

    def __init__(self, prefix: str = "rw"):
        self._prefix = prefix
        self._n = 0

    def new_id(self, kind: str) -> str:
        self._n += 1
        return f"{self._prefix}_{kind}_{self._n}"


@dataclass(frozen=True)
class RewriteProposal:
    kind: str
    delta: list[MemoryAtom]


def _candidate_bridge(q: str, *, z_q: np.ndarray, factory: AtomFactory, cfg: RewriteConfig) -> RewriteProposal:
    v_i = f"BRIDGE: restate query -> {q}"
    dim = int(z_q.shape[0])
    z_i = z_q.copy() if cfg.bridge_embedding == "from_q" else embed_text(v_i, dim=dim)
    atom = MemoryAtom(
        id=factory.new_id("bridge"),
        q_i=q,
        v_i=v_i,
        z_i=z_i,
        c_i=cfg.bridge_cost,
        s_i=cfg.bridge_valence,
        eta_i={"rewrite_type": "bridge"},
    )
    return RewriteProposal(kind="bridge", delta=[atom])


def _candidate_constraint(
    q: str,
    *,
    z_q: np.ndarray,
    M: Sequence[MemoryAtom],
    factory: AtomFactory,
    cfg: RewriteConfig,
) -> RewriteProposal | None:
    from .entropy import chi

    conflicts: list[tuple[str, str]] = []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            if chi(M[i], M[j], M=M) == 1:
                conflicts.append((M[i].id, M[j].id))

    if not conflicts:
        return None

    chosen = conflicts[: cfg.constraint_max_pairs]
    v_i = "CONSTRAINT: reconcile conflicting pair(s)"
    dim = int(z_q.shape[0])
    z_i = z_q.copy() if cfg.constraint_embedding == "from_q" else embed_text(v_i, dim=dim)
    atom = MemoryAtom(
        id=factory.new_id("constraint"),
        q_i=q,
        v_i=v_i,
        z_i=z_i,
        c_i=cfg.constraint_cost,
        s_i=cfg.constraint_valence,
        eta_i={"rewrite_type": "constraint", "reconcile_pairs": [list(p) for p in chosen]},
    )
    return RewriteProposal(kind="constraint", delta=[atom])


def _candidate_abstraction(
    q: str,
    *,
    M: Sequence[MemoryAtom],
    factory: AtomFactory,
    cfg: RewriteConfig,
) -> RewriteProposal | None:
    candidates = [m for m in M if not is_constraint_atom(m)]
    if len(candidates) < 2:
        return None

    best_pair: tuple[MemoryAtom, MemoryAtom] | None = None
    best_sim = -1.0
    for a, b in itertools.combinations(candidates, 2):
        sim = cos(a.z_i, b.z_i)
        if sim > best_sim:
            best_sim = sim
            best_pair = (a, b)

    if best_pair is None or best_sim < cfg.abstraction_similarity_threshold:
        return None

    a, b = best_pair
    v_i = f"ABSTRACT: combine {a.id} + {b.id}"
    if cfg.abstraction_embedding == "from_v":
        z_abs = embed_text(v_i, dim=int(a.z_i.shape[0]))
    else:
        z_abs = embed_average([a.z_i, b.z_i])
    sum_cost = float(a.c_i + b.c_i)
    atom = MemoryAtom(
        id=factory.new_id("abstract"),
        q_i=q,
        v_i=v_i,
        z_i=z_abs,
        c_i=min(cfg.abstraction_cost_ratio * sum_cost, sum_cost),
        s_i=cfg.abstraction_valence,
        eta_i={"rewrite_type": "abstraction", "subsumes": [a.id, b.id]},
    )
    return RewriteProposal(kind="abstraction", delta=[atom])


def propose_rewrite(
    q: str,
    *,
    z_q: np.ndarray,
    M: Sequence[MemoryAtom],
    entropy_before: EntropyBreakdown,
    history: HistoryStore,
    weights: EntropyWeights,
    rewrite: RewriteConfig,
    entropy_cfg: EntropyConfig,
    stability_cfg: StabilityConfig,
    factory: AtomFactory,
) -> tuple[RewriteProposal, EntropyBreakdown] | None:
    """Try a few candidate DeltaM and pick the best strict descent."""

    ordered_kinds = _ordered_kinds(entropy_before, rewrite=rewrite)
    evaluated = evaluate_rewrite_candidates(
        q,
        z_q=z_q,
        M=M,
        history=history,
        weights=weights,
        rewrite=rewrite,
        entropy_cfg=entropy_cfg,
        stability_cfg=stability_cfg,
        factory=factory,
        kinds=ordered_kinds,
    )

    best: tuple[RewriteProposal, EntropyBreakdown] | None = None
    for prop, after in evaluated:
        if after.total < entropy_before.total - 1e-12:
            if best is None or after.total < best[1].total:
                best = (prop, after)
    return best


def _load_templates(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = list(v)
    return out


def _bucket(entropy: EntropyBreakdown) -> str:
    parts = {"cov": entropy.E_cov, "conf": entropy.E_conf, "stab": entropy.E_stab}
    dominant = max(parts, key=parts.get)
    return f"dominant_{dominant}"


def _ordered_kinds(entropy: EntropyBreakdown, *, rewrite: RewriteConfig) -> list[str]:
    enabled = set(rewrite.enabled_kinds)
    templates = _load_templates(rewrite.template_path) if rewrite.mode == "template_then_search" else {}
    order = templates.get(_bucket(entropy), list(rewrite.enabled_kinds))

    ordered_kinds = [k for k in order if k in enabled]
    for k in rewrite.enabled_kinds:
        if k in enabled and k not in ordered_kinds:
            ordered_kinds.append(k)
    return ordered_kinds


def evaluate_rewrite_candidates(
    q: str,
    *,
    z_q: np.ndarray,
    M: Sequence[MemoryAtom],
    history: HistoryStore,
    weights: EntropyWeights,
    rewrite: RewriteConfig,
    entropy_cfg: EntropyConfig,
    stability_cfg: StabilityConfig,
    factory: AtomFactory,
    kinds: Sequence[str] | None = None,
) -> list[tuple[RewriteProposal, EntropyBreakdown]]:
    """Generate and score candidate rewrites, returning (proposal, entropy_after)."""

    enabled = set(rewrite.enabled_kinds)
    ordered = list(kinds) if kinds is not None else list(rewrite.enabled_kinds)
    ordered = [k for k in ordered if k in enabled]

    proposals: list[RewriteProposal] = []
    for kind in ordered:
        if kind == "bridge":
            proposals.append(_candidate_bridge(q, z_q=z_q, factory=factory, cfg=rewrite))
        elif kind == "constraint":
            maybe_constraint = _candidate_constraint(q, z_q=z_q, M=M, factory=factory, cfg=rewrite)
            if maybe_constraint is not None:
                proposals.append(maybe_constraint)
        elif kind == "abstraction":
            maybe_abs = _candidate_abstraction(q, M=M, factory=factory, cfg=rewrite)
            if maybe_abs is not None:
                proposals.append(maybe_abs)

    evaluated: list[tuple[RewriteProposal, EntropyBreakdown]] = []
    for prop in proposals:
        M2 = list(M) + list(prop.delta)
        after = compute_entropy(
            q,
            z_q=z_q,
            M=M2,
            history=history,
            weights=weights,
            coverage_clip_cosine=entropy_cfg.coverage_clip_cosine,
            conflict_include_constraints=entropy_cfg.conflict_include_constraints,
            cluster_tokens=stability_cfg.cluster_tokens,
        )
        evaluated.append((prop, after))
    return evaluated
