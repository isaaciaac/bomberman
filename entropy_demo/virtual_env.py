"""Virtual-environment training (minimal demo).

This module does not simulate a world. It generates entropy-evolution trajectories
so we can learn which rewrite *types* tend to reduce expressibility entropy under
different structural conditions.

Output is a small rewrite template file used by RewriteConfig.mode='template_then_search'.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG, ModelConfig
from .entropy import EntropyBreakdown, compute_entropy
from .memory_store import MemoryStore
from .retrieval import s1_retrieve
from .rewrite import AtomFactory, evaluate_rewrite_candidates
from .types import CognitiveState


@dataclass(frozen=True)
class VirtualEnvTrainConfig:
    episodes: int = 200
    exploration_prob: float = 0.2
    seed: int = 0
    output_path: Path = Path(".run/rewrite_templates.json")


def _bucket(entropy: EntropyBreakdown) -> str:
    parts = {"cov": entropy.E_cov, "conf": entropy.E_conf, "stab": entropy.E_stab}
    dominant = max(parts, key=parts.get)
    return f"dominant_{dominant}"


def train_rewrite_templates(
    queries: Sequence[str],
    *,
    model: ModelConfig = DEFAULT_CONFIG,
    env: VirtualEnvTrainConfig = VirtualEnvTrainConfig(),
) -> dict[str, list[str]]:
    """Train a simple bucket->rewrite-order template mapping."""

    if not queries:
        raise ValueError("queries must be non-empty")

    rng = random.Random(env.seed)
    store = MemoryStore(seed_path=model.seed_path, embedding=model.embedding, writeback=model.writeback)
    store.load()

    # Aggregate immediate entropy reduction per (bucket, rewrite_kind).
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}

    for _ in range(env.episodes):
        q = rng.choice(list(queries))
        retrieval = s1_retrieve(
            q,
            store.atoms,
            k=model.retrieval.k,
            dim=model.embedding.dim,
            token_min_len=model.embedding.token_min_len,
            use_english_stopwords=model.embedding.use_english_stopwords,
            valence_mode=model.retrieval.valence_mode,
            score_threshold=model.retrieval.score_threshold,
        )
        z_q = retrieval.z_q
        state = CognitiveState(q=q, M=tuple(retrieval.M0))

        # Use a fresh factory per episode to keep ids readable.
        factory = AtomFactory(prefix="ve")

        entropy = compute_entropy(
            q,
            z_q=z_q,
            M=state.M,
            history=_NullHistory(model.stability.history.p0),
            weights=model.entropy.weights,
            coverage_clip_cosine=model.entropy.coverage_clip_cosine,
            conflict_include_constraints=model.entropy.conflict_include_constraints,
            cluster_tokens=model.stability.cluster_tokens,
        )

        for _t in range(model.rewrite.tmax):
            if entropy.total <= model.entropy.epsilon:
                break

            evaluated = evaluate_rewrite_candidates(
                q,
                z_q=z_q,
                M=state.M,
                history=_NullHistory(model.stability.history.p0),
                weights=model.entropy.weights,
                rewrite=model.rewrite,
                entropy_cfg=model.entropy,
                stability_cfg=model.stability,
                factory=factory,
            )
            descending = [(p, e) for (p, e) in evaluated if e.total < entropy.total - 1e-12]
            if not descending:
                break

            if rng.random() < env.exploration_prob:
                chosen, after = rng.choice(descending)
            else:
                chosen, after = min(descending, key=lambda x: x[1].total)

            b = _bucket(entropy)
            totals.setdefault(b, {}).setdefault(chosen.kind, 0.0)
            counts.setdefault(b, {}).setdefault(chosen.kind, 0)
            totals[b][chosen.kind] += float(entropy.total - after.total)
            counts[b][chosen.kind] += 1

            state = state.with_added(chosen.delta)
            entropy = after

    templates: dict[str, list[str]] = {}
    for bucket, per_kind in totals.items():
        scored: list[tuple[float, str]] = []
        for kind, total_delta in per_kind.items():
            n = counts.get(bucket, {}).get(kind, 0)
            if n <= 0:
                continue
            scored.append((total_delta / n, kind))
        scored.sort(reverse=True)
        templates[bucket] = [kind for _, kind in scored]

    env.output_path.parent.mkdir(parents=True, exist_ok=True)
    env.output_path.write_text(json.dumps(templates, indent=2, sort_keys=True), encoding="utf-8")
    return templates


class _NullHistory:
    """Minimal HistoryStore-compatible object for training (no persistence)."""

    def __init__(self, p0: float):
        self._p0 = float(p0)

    @property
    def config(self):
        # Only fields used by compute_entropy() for sig; defaults are fine.
        from .entropy import HistoryConfig

        return HistoryConfig(p0=self._p0)

    def get_p_succ(self, *, cluster: str, sig: str) -> float:
        return self._p0
