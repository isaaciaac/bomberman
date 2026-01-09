"""Offline, deterministic baseline: S1-only decision with citations.

This is *not* a Transformer. It is a minimal baseline you can run locally to
compare against the entropy-gated cognitive core:

  1) Run this to produce baseline outputs JSONL.
  2) Score it via `python -m entropy_demo.eval --baseline-jsonl ...`.

The baseline:
- Sees only S1 retrieved memories (no S2).
- Decides answer/refuse using simple, auditable thresholds.
- If it answers, it must cite `used_ids` from the retrieved set.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG, load_model_config, model_config_from_dict, model_config_to_dict
from .embedding import cos, embed_text, tokenize
from .entropy import chi
from .memory_store import MemoryStore
from .retrieval import s1_retrieve
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


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return override


def _load_tasks(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("tasks file must be a JSON object")
    return raw


def _required_claim_keys_from_query(q: str) -> list[str]:
    keys: list[str] = []
    for chunk in _REQ_RE.findall(q):
        for part in chunk.split(","):
            k = part.strip()
            if k:
                keys.append(k)
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _atom_claim_key(atom: MemoryAtom) -> str | None:
    parsed = atom.eta_i.get("parsed") or {}
    key = parsed.get("key")
    if isinstance(key, str) and key:
        return key
    ck = atom.eta_i.get("claim_key", atom.eta_i.get("key"))
    if isinstance(ck, str) and ck:
        return ck
    return None


def _content_tokens(text: str, *, token_min_len: int, use_en_stopwords: bool) -> set[str]:
    toks = tokenize(text)
    out: set[str] = set()
    for t in toks:
        if len(t) < token_min_len:
            continue
        if use_en_stopwords and t in _EN_STOPWORDS:
            continue
        out.add(t)
    return out


def _evidence_token_overlap_max(
    q: str,
    M0: list[MemoryAtom],
    *,
    token_min_len: int,
    use_en_stopwords: bool,
) -> int:
    q_toks = _content_tokens(q, token_min_len=token_min_len, use_en_stopwords=use_en_stopwords)
    best = 0
    for m in M0:
        overlap = len(q_toks.intersection(_content_tokens(m.v_i, token_min_len=token_min_len, use_en_stopwords=use_en_stopwords)))
        best = max(best, int(overlap))
    return int(best)


def _sim_max(q: str, M0: list[MemoryAtom], *, dim: int, clip: bool) -> tuple[float, dict[str, float]]:
    z_q = embed_text(q, dim=dim)
    sims: dict[str, float] = {}
    for m in M0:
        if m.z_i.shape != z_q.shape:
            continue
        s = cos(z_q, m.z_i)
        if clip:
            s = max(0.0, float(s))
        sims[m.id] = float(s)
    return (max(sims.values()) if sims else 0.0), sims


def _has_unreconciled_conflicts(M0: list[MemoryAtom]) -> bool:
    items = [m for m in M0 if not is_constraint_atom(m)]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if chi(items[i], items[j], M=M0) == 1:
                return True
    return False


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(r), ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline S1-only baseline for entropy_demo tasks")
    parser.add_argument("--tasks", type=str, default="data/tasks.json", help="Tasks JSON path")
    parser.add_argument("--config", type=str, default="", help="Optional base config JSON path")
    parser.add_argument("--mode", choices=["isolated", "persistent"], default="isolated")
    parser.add_argument("--out", type=str, default=".run/baseline_heuristic_outputs.jsonl", help="Output baseline JSONL path")

    parser.add_argument("--min-sim", type=float, default=0.25, help="Answer if sim_max >= this threshold")
    parser.add_argument("--clip-cosine", action="store_true", help="Clip cosine to [0,1] for similarity checks")
    parser.add_argument("--min-token-overlap", type=int, default=1, help="Answer if max token overlap >= this threshold")
    parser.add_argument("--token-min-len", type=int, default=3, help="Minimum token length for overlap")
    parser.add_argument("--use-en-stopwords", action="store_true", help="Filter common English stopwords for overlap")
    parser.add_argument("--require-no-conflicts", action="store_true", help="Refuse if any unreconciled conflicts exist in M0")
    parser.add_argument(
        "--require-claim-keys",
        action="store_true",
        help="If query contains REQ[...], refuse unless those claim keys appear in M0",
    )

    args = parser.parse_args()

    tasks_doc = _load_tasks(Path(args.tasks))
    tasks = tasks_doc.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("tasks.json must contain a non-empty 'tasks' list")

    base_cfg = load_model_config(Path(args.config)) if args.config else DEFAULT_CONFIG

    # Mirror eval defaults for fair comparison on retrieval.
    if not args.config:
        defaults = tasks_doc.get("recommended_defaults", {})
        if isinstance(defaults, dict):
            k = defaults.get("retrieval_k")
            if isinstance(k, int):
                base_cfg = replace(base_cfg, retrieval=replace(base_cfg.retrieval, k=int(k)))

    if args.mode == "isolated":
        base_cfg = replace(
            base_cfg,
            writeback=replace(base_cfg.writeback, enabled=False),
            capacity=replace(base_cfg.capacity, enabled=False),
        )

    store = MemoryStore(seed_path=base_cfg.seed_path, embedding=base_cfg.embedding, writeback=base_cfg.writeback)
    store.load()

    outputs: list[dict[str, Any]] = []

    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", ""))
        q = str(t.get("query", ""))
        overrides = t.get("task_overrides", {})
        if not tid or not q:
            continue

        cfg = base_cfg
        if isinstance(overrides, dict) and overrides:
            merged = model_config_to_dict(base_cfg)
            merged = _deep_merge(merged, overrides)
            cfg = model_config_from_dict(merged)
            if args.mode == "isolated":
                cfg = replace(cfg, writeback=replace(cfg.writeback, enabled=False), capacity=replace(cfg.capacity, enabled=False))

        ret = s1_retrieve(
            q,
            store.atoms,
            k=cfg.retrieval.k,
            dim=cfg.embedding.dim,
            token_min_len=cfg.embedding.token_min_len,
            use_english_stopwords=cfg.embedding.use_english_stopwords,
            valence_mode=cfg.retrieval.valence_mode,
            score_threshold=cfg.retrieval.score_threshold,
        )
        M0 = list(ret.M0)

        sim_max, sims = _sim_max(q, M0, dim=cfg.embedding.dim, clip=bool(args.clip_cosine))
        overlap_max = _evidence_token_overlap_max(
            q,
            M0,
            token_min_len=int(args.token_min_len),
            use_en_stopwords=bool(args.use_en_stopwords),
        )
        has_conflicts = _has_unreconciled_conflicts(M0) if bool(args.require_no_conflicts) else False

        required_keys = _required_claim_keys_from_query(q) if bool(args.require_claim_keys) else []
        present_keys = {k for m in M0 if (k := _atom_claim_key(m)) is not None}
        missing_keys = [k for k in required_keys if k not in present_keys]

        refuse_reasons: list[str] = []
        if sim_max < float(args.min_sim):
            refuse_reasons.append(f"sim_max<{float(args.min_sim):.3f} (got {sim_max:.3f})")
        if overlap_max < int(args.min_token_overlap):
            refuse_reasons.append(f"token_overlap_max<{int(args.min_token_overlap)} (got {overlap_max})")
        if has_conflicts:
            refuse_reasons.append("unreconciled_conflicts_in_M0")
        if missing_keys:
            refuse_reasons.append(f"missing_required_claim_keys={missing_keys}")

        refuse = len(refuse_reasons) > 0

        used_ids: list[str] = []
        if not refuse and sims:
            used_ids = [k for k, _ in sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[:2]]

        outputs.append(
            {
                "task_id": tid,
                "refuse": bool(refuse),
                "answer": "" if refuse else "BASELINE: answered from S1-only retrieval.",
                "reason": "; ".join(refuse_reasons),
                "used_ids": used_ids,
                "metrics": {"sim_max": float(sim_max), "token_overlap_max": int(overlap_max)},
            }
        )

    out_path = Path(args.out)
    _write_jsonl(out_path, outputs)
    print(json.dumps({"tasks": len(outputs), "out": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
