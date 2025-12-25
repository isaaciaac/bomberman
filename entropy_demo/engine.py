"""End-to-end demo engine: S1 -> entropy gate -> S2 REWRITE loop.

This module implements the Cognitive Core (Layer 1) orchestration:
S1 retrieval, expressibility entropy gate, and the S2 rewrite loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .embedding import embed_text
from .entropy import (
    EntropyBreakdown,
    EntropyWeights,
    HistoryConfig,
    HistoryStore,
    compute_entropy,
    enrich_eta_from_v,
)
from .retrieval import s1_retrieve
from .rewrite import AtomFactory, propose_rewrite, soft_fold_if_over_budget
from .types import CognitiveState, MemoryAtom, ensure_eta_dict


@dataclass(frozen=True)
class EngineConfig:
    k: int = 5
    epsilon: float = 0.35
    tmax: int = 6
    dim: int = 256
    weights: EntropyWeights = EntropyWeights()
    history: HistoryConfig = HistoryConfig()
    history_path: Path = Path(".run/history.json")
    c_max: float = 50.0


@dataclass(frozen=True)
class StepRecord:
    t: int
    M_size: int
    entropy: EntropyBreakdown
    rewrite_kind: str
    added_ids: list[str]


@dataclass(frozen=True)
class EngineResult:
    success: bool
    reason: str
    initial_state: CognitiveState
    final_state: CognitiveState
    initial_entropy: EntropyBreakdown
    final_entropy: EntropyBreakdown
    steps: list[StepRecord]


def _load_seed_memory(path: Path, *, dim: int) -> list[MemoryAtom]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[MemoryAtom] = []
    for item in raw:
        atom = MemoryAtom(
            id=str(item["id"]),
            q_i=str(item.get("q_i", "")),
            v_i=str(item["v_i"]),
            z_i=embed_text(str(item["v_i"]), dim=dim),
            c_i=float(item.get("c_i", 0.0)),
            s_i=float(item.get("s_i", 0.0)),
            eta_i=ensure_eta_dict(item.get("eta_i")),
        )
        enrich_eta_from_v(atom)
        out.append(atom)
    return out


def run_engine(
    q: str,
    *,
    seed_path: Path,
    config: EngineConfig = EngineConfig(),
    verbose: bool = True,
) -> EngineResult:
    """Run the Cognitive Core loop and return an EngineResult.

    This function decides whether the system is allowed to enter expression.
    It does not generate an answer string or influence the entropy dynamics.
    """

    memory_store = _load_seed_memory(seed_path, dim=config.dim)
    history = HistoryStore(config.history_path, config=config.history)
    history.load()

    retrieval = s1_retrieve(q, memory_store, k=config.k, dim=config.dim)
    z_q = retrieval.z_q
    initial_state = CognitiveState(q=q, M=tuple(retrieval.M0))
    state = initial_state

    ent0 = compute_entropy(q, z_q=z_q, M=state.M, history=history, weights=config.weights)
    steps: list[StepRecord] = []

    if verbose:
        print(f"S1: retrieved |M0|={len(state.M)} (k={config.k}) ids={state.ids}")
        print(
            "t=0 |M|={:d} E={:.4f} (E_cov={:.4f} E_conf={:.4f} E_stab={:.4f} p_succ={:.2f})".format(
                len(state.M), ent0.total, ent0.E_cov, ent0.E_conf, ent0.E_stab, ent0.p_succ
            )
        )

    if ent0.total <= config.epsilon:
        history.update(cluster=ent0.cluster, sig=ent0.sig, success=True)
        return EngineResult(
            success=True,
            reason="E(q,M0) <= epsilon",
            initial_state=initial_state,
            final_state=state,
            initial_entropy=ent0,
            final_entropy=ent0,
            steps=steps,
        )

    factory = AtomFactory(prefix="rw")
    current = ent0

    for t in range(1, config.tmax + 1):
        decision = propose_rewrite(
            q,
            z_q=z_q,
            M=state.M,
            entropy_before=current,
            history=history,
            weights=config.weights,
            factory=factory,
        )
        if decision is None:
            reason = "cannot reduce entropy under constraints (no admissible REWRITE)"
            if verbose:
                print(f"STOP: {reason}")
            history.update(cluster=current.cluster, sig=current.sig, success=False)
            return EngineResult(
                success=False,
                reason=reason,
                initial_state=initial_state,
                final_state=state,
                initial_entropy=ent0,
                final_entropy=current,
                steps=steps,
            )

        proposal, after = decision
        added = proposal.delta
        added_ids = [a.id for a in added]

        # No-DROP monotone extension: M_{t+1} = M_t U DeltaM_t
        state = state.with_added(added)
        for a in added:
            memory_store.append(a)

        current = after
        steps.append(
            StepRecord(
                t=t,
                M_size=len(state.M),
                entropy=current,
                rewrite_kind=proposal.kind,
                added_ids=added_ids,
            )
        )

        if verbose:
            print(
                "t={:d} |M|={:d} E={:.4f} (E_cov={:.4f} E_conf={:.4f} E_stab={:.4f}) rewrite={} added={}".format(
                    t,
                    len(state.M),
                    current.total,
                    current.E_cov,
                    current.E_conf,
                    current.E_stab,
                    proposal.kind,
                    added_ids,
                )
            )

        # Optional budget folding (demo approximation).
        folded = soft_fold_if_over_budget(memory_store, c_max=config.c_max, factory=factory)
        if verbose and folded is not None:
            print(f"FOLD: added {folded.id} to respect C_max~{config.c_max} (soft)")

        if current.total <= config.epsilon:
            history.update(cluster=current.cluster, sig=current.sig, success=True)
            return EngineResult(
                success=True,
                reason="E(q,M*) <= epsilon",
                initial_state=initial_state,
                final_state=state,
                initial_entropy=ent0,
                final_entropy=current,
                steps=steps,
            )

    reason = f"reached T_max={config.tmax} without entering R_epsilon"
    if verbose:
        print(f"STOP: {reason}")
    history.update(cluster=current.cluster, sig=current.sig, success=False)
    return EngineResult(
        success=False,
        reason=reason,
        initial_state=initial_state,
        final_state=state,
        initial_entropy=ent0,
        final_entropy=current,
        steps=steps,
    )
