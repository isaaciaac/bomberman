"""Batch runner for verifying the cognitive core on multiple queries."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, load_model_config, model_config_to_dict
from .engine import run_engine


def _load_queries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    queries: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        queries.append(s)
    return queries


def _entropy_to_dict(e: Any) -> dict[str, Any]:
    return {
        "total": float(e.total),
        "E_cov": float(e.E_cov),
        "E_conf": float(e.E_conf),
        "E_stab": float(e.E_stab),
        "sim_max": float(e.sim_max),
        "p_succ": float(e.p_succ),
        "cluster": str(e.cluster),
        "sig": str(e.sig),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch evaluation for entropy_demo")
    parser.add_argument("--queries", required=True, help="Path to a text file with one query per line")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config path")
    parser.add_argument("--out", type=str, default=".run/batch_results.json", help="Output JSON path")
    parser.add_argument(
        "--mode",
        choices=["isolated", "persistent"],
        default="isolated",
        help="isolated: do not write history/write-back; persistent: use config persistence",
    )
    args = parser.parse_args()

    cfg = load_model_config(Path(args.config)) if args.config else DEFAULT_CONFIG
    queries = _load_queries(Path(args.queries))
    if not queries:
        raise SystemExit("no queries found")

    tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.mode == "isolated":
        tmp = tempfile.TemporaryDirectory()
        td = Path(tmp.name)
        cfg = replace(
            cfg,
            stability=replace(cfg.stability, history_path=td / "history.json"),
            writeback=replace(cfg.writeback, enabled=False),
            capacity=replace(cfg.capacity, enabled=False),
        )

    runs: list[dict[str, Any]] = []
    success_count = 0
    steps_success_total = 0
    rewrite_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

    for q in queries:
        res = run_engine(q, config=cfg, verbose=False)
        success_count += 1 if res.success else 0
        reason_counts[res.reason] = int(reason_counts.get(res.reason, 0)) + 1
        for step in res.steps:
            rewrite_counts[step.rewrite_kind] = int(rewrite_counts.get(step.rewrite_kind, 0)) + 1
        if res.success:
            steps_success_total += len(res.steps)

        runs.append(
            {
                "query": q,
                "success": bool(res.success),
                "reason": str(res.reason),
                "steps": int(len(res.steps)),
                "rewrites": [s.rewrite_kind for s in res.steps],
                "entropy": {
                    "initial": _entropy_to_dict(res.initial_entropy),
                    "final": _entropy_to_dict(res.final_entropy),
                },
            }
        )

    summary = {
        "n": int(len(runs)),
        "success": int(success_count),
        "success_rate": float(success_count / len(runs)),
        "mean_steps_success": float(steps_success_total / success_count) if success_count else None,
        "rewrite_counts": dict(sorted(rewrite_counts.items(), key=lambda x: (-x[1], x[0]))),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))),
    }

    payload = {
        "mode": args.mode,
        "config": model_config_to_dict(cfg),
        "summary": summary,
        "runs": runs,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

