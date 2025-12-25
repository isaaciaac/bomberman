"""S2 rewrite proposals that monotonically extend M and strictly decrease entropy."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

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


def _candidate_bridge(q: str, *, z_q: np.ndarray, factory: AtomFactory) -> RewriteProposal:
    atom = MemoryAtom(
        id=factory.new_id("bridge"),
        q_i=q,
        v_i=f"BRIDGE: restate query -> {q}",
        z_i=z_q.copy(),
        c_i=0.1,
        s_i=0.6,
        eta_i={"rewrite_type": "bridge"},
    )
    return RewriteProposal(kind="bridge", delta=[atom])


def _candidate_constraint(
    q: str,
    *,
    z_q: np.ndarray,
    M: Sequence[MemoryAtom],
    factory: AtomFactory,
    max_pairs: int = 3,
) -> RewriteProposal | None:
    from .entropy import chi

    conflicts: list[tuple[str, str]] = []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            if chi(M[i], M[j], M=M) == 1:
                conflicts.append((M[i].id, M[j].id))

    if not conflicts:
        return None

    chosen = conflicts[:max_pairs]
    atom = MemoryAtom(
        id=factory.new_id("constraint"),
        q_i=q,
        v_i="CONSTRAINT: reconcile conflicting pair(s)",
        # Use the query embedding so this candidate mediates conflict without
        # worsening coverage; in conflict-free cases the bridge candidate wins.
        z_i=z_q.copy(),
        c_i=0.05,
        s_i=0.3,
        eta_i={"rewrite_type": "constraint", "reconcile_pairs": [list(p) for p in chosen]},
    )
    return RewriteProposal(kind="constraint", delta=[atom])


def _candidate_abstraction(
    q: str,
    *,
    M: Sequence[MemoryAtom],
    factory: AtomFactory,
    similarity_threshold: float = 0.8,
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

    if best_pair is None or best_sim < similarity_threshold:
        return None

    a, b = best_pair
    z_abs = embed_average([a.z_i, b.z_i])
    sum_cost = float(a.c_i + b.c_i)
    atom = MemoryAtom(
        id=factory.new_id("abstract"),
        q_i=q,
        v_i=f"ABSTRACT: {a.id}+{b.id} (sim={best_sim:.2f})",
        z_i=z_abs,
        c_i=min(0.5 * sum_cost, sum_cost),
        s_i=0.5,
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
    factory: AtomFactory,
) -> tuple[RewriteProposal, EntropyBreakdown] | None:
    """Try a few candidate DeltaM and pick the best strict descent."""

    proposals: list[RewriteProposal] = []
    proposals.append(_candidate_bridge(q, z_q=z_q, factory=factory))

    maybe_constraint = _candidate_constraint(q, z_q=z_q, M=M, factory=factory)
    if maybe_constraint is not None:
        proposals.append(maybe_constraint)

    maybe_abs = _candidate_abstraction(q, M=M, factory=factory)
    if maybe_abs is not None:
        proposals.append(maybe_abs)

    best: tuple[RewriteProposal, EntropyBreakdown] | None = None
    for prop in proposals:
        M2 = list(M) + list(prop.delta)
        after = compute_entropy(q, z_q=z_q, M=M2, history=history, weights=weights)
        if after.total < entropy_before.total - 1e-12:
            if best is None or after.total < best[1].total:
                best = (prop, after)
    return best


def active_cost(memory_store: Sequence[MemoryAtom], *, s_min: float = 0.05) -> float:
    """Approximate active memory cost used for the demo's folding heuristic."""

    subsumed: set[str] = set()
    for m in memory_store:
        subsumes = m.eta_i.get("subsumes", [])
        if isinstance(subsumes, list):
            subsumed.update(str(x) for x in subsumes)

    return float(
        sum(
            m.c_i
            for m in memory_store
            if m.id not in subsumed and max(0.0, float(m.s_i)) >= s_min
        )
    )


def soft_fold_if_over_budget(
    memory_store: list[MemoryAtom],
    *,
    c_max: float,
    factory: AtomFactory,
) -> MemoryAtom | None:
    """Soft folding: add an abstraction atom without mutating existing atoms.

    Demo approximation:
    - No deletion.
    - No in-place modification of existing atoms.
    - Budget is enforced by treating atoms referenced by an `ABSTRACT:` atom's
      `eta_i.subsumes` list as inactive for cost accounting.
    """

    if active_cost(memory_store) <= c_max:
        return None

    subsumed: set[str] = set()
    for m in memory_store:
        subsumes = m.eta_i.get("subsumes", [])
        if isinstance(subsumes, list):
            subsumed.update(str(x) for x in subsumes)

    candidates = [
        m
        for m in memory_store
        if not is_constraint_atom(m)
        and max(0.0, float(m.s_i)) >= 0.05
        and m.id not in subsumed
    ]
    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda m: float(m.s_i))
    a, b = candidates[0], candidates[1]

    z_abs = embed_average([a.z_i, b.z_i])
    sum_cost = float(a.c_i + b.c_i)
    new_atom = MemoryAtom(
        id=factory.new_id("fold"),
        q_i="folding",
        v_i=f"ABSTRACT: fold {a.id}+{b.id} (budget)",
        z_i=z_abs,
        c_i=min(0.25 * sum_cost, sum_cost),
        s_i=0.4,
        eta_i={"rewrite_type": "fold", "subsumes": [a.id, b.id]},
    )

    memory_store.append(new_atom)
    return new_atom
