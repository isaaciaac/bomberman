"""Generate a prompt pack for baseline (e.g., Transformer) comparison.

This module does not run any Transformer. It prepares deterministic inputs so you
can run an external model (offline or elsewhere) and then score it via:

  python -m entropy_demo.eval --tasks data/tasks.json --baseline-jsonl <outputs.jsonl>

The baseline output format expected by the evaluator is JSONL with at least:
  {"task_id": "...", "refuse": true|false, "used_ids": ["m003", ...], ...}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG, load_model_config, model_config_from_dict, model_config_to_dict
from .memory_store import MemoryStore
from .reporting import memory_atom_to_dict
from .retrieval import s1_retrieve


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


def _make_prompt(*, task_id: str, query: str, atoms: list[dict[str, Any]]) -> str:
    # Keep this short and rigid so models are more likely to output valid JSON.
    lines: list[str] = []
    lines.append("You are given a query and a small set of memory atoms (each has an id and text).")
    lines.append("Decide whether you can answer using ONLY these memories.")
    lines.append("")
    lines.append("Rules:")
    lines.append("- If memories are insufficient, set refuse=true.")
    lines.append("- If you answer, do not invent facts; cite which memory ids you used.")
    lines.append("- Output MUST be a single JSON object, nothing else.")
    lines.append("")
    lines.append("Output JSON schema:")
    lines.append('{ "task_id": "<id>", "refuse": true|false, "answer": "<string or empty>", "reason": "<string or empty>", "used_ids": ["m001", "..."] }')
    lines.append("")
    lines.append("Input:")
    lines.append(f'task_id: "{task_id}"')
    lines.append(f'query: "{query}"')
    lines.append("memories:")
    for a in atoms:
        mid = str(a.get("id", ""))
        v = str(a.get("v_i", ""))
        lines.append(f'- {mid}: {v}')
    return "\n".join(lines)


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(r), ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline prompt pack (Transformer comparison helper)")
    parser.add_argument("--tasks", type=str, default="data/tasks.json", help="Tasks JSON path")
    parser.add_argument("--config", type=str, default="", help="Optional base config JSON path")
    parser.add_argument("--mode", choices=["isolated", "persistent"], default="isolated")
    parser.add_argument("--out", type=str, default=".run/baseline_prompt_pack.jsonl", help="Output prompt-pack JSONL path")
    parser.add_argument(
        "--template-out",
        type=str,
        default=".run/baseline_outputs_template.jsonl",
        help="Output baseline-outputs JSONL template path",
    )
    args = parser.parse_args()

    tasks_doc = _load_tasks(Path(args.tasks))
    tasks = tasks_doc.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("tasks.json must contain a non-empty 'tasks' list")

    base_cfg = load_model_config(Path(args.config)) if args.config else DEFAULT_CONFIG

    # Mirror eval.py behavior: apply recommended defaults when no config is provided.
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

    pack: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []

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

        retrieval = s1_retrieve(
            q,
            store.atoms,
            k=cfg.retrieval.k,
            dim=cfg.embedding.dim,
            token_min_len=cfg.embedding.token_min_len,
            use_english_stopwords=cfg.embedding.use_english_stopwords,
            valence_mode=cfg.retrieval.valence_mode,
            score_threshold=cfg.retrieval.score_threshold,
        )

        atoms = [memory_atom_to_dict(m) for m in retrieval.M0]
        prompt = _make_prompt(task_id=tid, query=q, atoms=atoms)

        pack.append(
            {
                "task_id": tid,
                "query": q,
                "s1": {"k": int(cfg.retrieval.k), "retrieved_ids": [a["id"] for a in atoms], "retrieved_atoms": atoms},
                "prompt": prompt,
            }
        )
        template.append({"task_id": tid, "refuse": None, "answer": "", "reason": "", "used_ids": []})

    _write_jsonl(Path(args.out), pack)
    _write_jsonl(Path(args.template_out), template)
    print(json.dumps({"prompts": len(pack), "out": args.out, "template_out": args.template_out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
