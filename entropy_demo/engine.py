"""Cognitive core loop: S1 -> entropy gate -> S2 rewrite loop.

This module implements the Layer 1 logic described in the paper:

- State is (q, M) and excludes answer text.
- S1 retrieves an initial candidate set M0 from the long-term store calM.
- The entropy discriminator decides whether expression is allowed.
- If not expressible, S2 performs REWRITE steps that *only add* new structure
  and must strictly decrease E(q, M).

Language generation/projection is intentionally not part of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG, ModelConfig
from .entropy import (
    EntropyBreakdown,
    HistoryStore,
    compute_entropy,
)
from .memory_store import MemoryStore
from .retrieval import s1_retrieve
from .rewrite import AtomFactory, propose_rewrite
from .types import CognitiveState, MemoryAtom
from .verifier import VerifierResult, verify_state


@dataclass(frozen=True)
class StepRecord:
    t: int
    M_size: int
    entropy: EntropyBreakdown
    verifier_passed: bool
    verifier_reasons: tuple[str, ...]
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
    verifier_initial: VerifierResult
    verifier_final: VerifierResult
    steps: list[StepRecord]
    writeback_persisted: list[MemoryAtom]
    capacity_created: list[MemoryAtom]


def run_engine(
    q: str,
    *,
    config: ModelConfig = DEFAULT_CONFIG,
    seed_path: Path | None = None,
    verbose: bool = True,
) -> EngineResult:
    """Run the Cognitive Core loop and return an EngineResult.

    This function decides whether the system is allowed to enter expression.
    It does not generate an answer string or influence the entropy dynamics.
    """

    store = MemoryStore(seed_path=seed_path or config.seed_path, embedding=config.embedding, writeback=config.writeback)
    store.load()

    history = HistoryStore(config.stability.history_path, config=config.stability.history)
    history.load()

    retrieval = s1_retrieve(
        q,
        store.atoms,
        k=config.retrieval.k,
        dim=config.embedding.dim,
        token_min_len=config.embedding.token_min_len,
        use_english_stopwords=config.embedding.use_english_stopwords,
        valence_mode=config.retrieval.valence_mode,
        score_threshold=config.retrieval.score_threshold,
    )
    z_q = retrieval.z_q
    initial_state = CognitiveState(q=q, M=tuple(retrieval.M0))
    state = initial_state

    ent0 = compute_entropy(
        q,
        z_q=z_q,
        M=state.M,
        history=history,
        weights=config.entropy.weights,
        coverage_clip_cosine=config.entropy.coverage_clip_cosine,
        conflict_include_constraints=config.entropy.conflict_include_constraints,
        cluster_tokens=config.stability.cluster_tokens,
    )
    steps: list[StepRecord] = []
    v0 = verify_state(q, state.M, cfg=config.verifier, z_q=z_q)

    if verbose:
        print(f"S1: retrieved |M0|={len(state.M)} (k={config.retrieval.k}) ids={state.ids}")
        print(
            "t=0 |M|={:d} E={:.4f} (E_cov={:.4f} E_conf={:.4f} E_stab={:.4f} p_succ={:.2f})".format(
                len(state.M), ent0.total, ent0.E_cov, ent0.E_conf, ent0.E_stab, ent0.p_succ
            )
        )
        if config.verifier.enabled:
            print(f"V: passed={v0.passed} reasons={list(v0.reasons)}")

    if ent0.total <= config.entropy.epsilon and v0.passed:
        history.update(cluster=ent0.cluster, sig=ent0.sig, success=True)
        persisted = store.write_back([], success=True)
        created = store.enforce_capacity(capacity=config.capacity)
        return EngineResult(
            success=True,
            reason="E(q,M0) <= epsilon and verifier passed",
            initial_state=initial_state,
            final_state=state,
            initial_entropy=ent0,
            final_entropy=ent0,
            verifier_initial=v0,
            verifier_final=v0,
            steps=steps,
            writeback_persisted=persisted,
            capacity_created=created,
        )

    factory = AtomFactory(prefix="rw")
    current = ent0
    current_v = v0

    for t in range(1, config.rewrite.tmax + 1):
        # If already within expressible region but verifier fails, we continue
        # searching for a strictly entropy-decreasing rewrite that also moves
        # the state toward acceptance.
        decision = propose_rewrite(
            q,
            z_q=z_q,
            M=state.M,
            entropy_before=current,
            history=history,
            weights=config.entropy.weights,
            rewrite=config.rewrite,
            entropy_cfg=config.entropy,
            stability_cfg=config.stability,
            factory=factory,
        )
        if decision is None:
            reason = "cannot reduce entropy under constraints (no admissible REWRITE)"
            if current.total <= config.entropy.epsilon and not current_v.passed:
                reason = "verifier failed and no admissible entropy-decreasing REWRITE exists"
            if verbose:
                print(f"STOP: {reason}")
            history.update(cluster=current.cluster, sig=current.sig, success=False)
            persisted = store.write_back([], success=False)
            created = store.enforce_capacity(capacity=config.capacity)
            return EngineResult(
                success=False,
                reason=reason,
                initial_state=initial_state,
                final_state=state,
                initial_entropy=ent0,
                final_entropy=current,
                verifier_initial=v0,
                verifier_final=current_v,
                steps=steps,
                writeback_persisted=persisted,
                capacity_created=created,
            )

        proposal, after = decision
        added = proposal.delta
        added_ids = [a.id for a in added]

        # No-DROP monotone extension: M_{t+1} = M_t U DeltaM_t
        state = state.with_added(added)
        current = after
        current_v = verify_state(q, state.M, cfg=config.verifier, z_q=z_q)
        steps.append(
            StepRecord(
                t=t,
                M_size=len(state.M),
                entropy=current,
                verifier_passed=current_v.passed,
                verifier_reasons=tuple(current_v.reasons),
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
            if config.verifier.enabled:
                print(f"V: passed={current_v.passed} reasons={list(current_v.reasons)}")

        if current.total <= config.entropy.epsilon and current_v.passed:
            history.update(cluster=current.cluster, sig=current.sig, success=True)
            delta_total = [m for m in state.M if m.id not in {x.id for x in initial_state.M}]
            persisted = store.write_back(delta_total, success=True)
            created = store.enforce_capacity(capacity=config.capacity)
            return EngineResult(
                success=True,
                reason="E(q,M*) <= epsilon and verifier passed",
                initial_state=initial_state,
                final_state=state,
                initial_entropy=ent0,
                final_entropy=current,
                verifier_initial=v0,
                verifier_final=current_v,
                steps=steps,
                writeback_persisted=persisted,
                capacity_created=created,
            )

    reason = f"reached T_max={config.rewrite.tmax} without entering R_epsilon"
    if current.total <= config.entropy.epsilon and not current_v.passed:
        reason = f"reached T_max={config.rewrite.tmax} with verifier still failing"
    if verbose:
        print(f"STOP: {reason}")
    history.update(cluster=current.cluster, sig=current.sig, success=False)
    persisted = store.write_back([], success=False)
    created = store.enforce_capacity(capacity=config.capacity)
    return EngineResult(
        success=False,
        reason=reason,
        initial_state=initial_state,
        final_state=state,
        initial_entropy=ent0,
        final_entropy=current,
        verifier_initial=v0,
        verifier_final=current_v,
        steps=steps,
        writeback_persisted=persisted,
        capacity_created=created,
    )
