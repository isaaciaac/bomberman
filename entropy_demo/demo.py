"""End-to-end demo runner that produces a compact engineering trace.

This module is intentionally "not pretty": it aims to show the full lifecycle:
- S1 retrieval
- S2 activation (if needed)
- projection using multiple memories
- optional write-back persistence
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, load_model_config, model_config_to_dict
from .engine import run_engine
from .generate import generate_text
from .reporting import engine_result_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end demo (engineering trace)")
    parser.add_argument("--query", required=True, help="Input query/signal string")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config path")
    parser.add_argument("--out", type=str, default=".run/demo_trace.json", help="Output JSON path")
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of repeated runs (useful to trigger write-back when min_successes>1)",
    )
    parser.add_argument(
        "--mode",
        choices=["persistent", "isolated"],
        default="persistent",
        help="isolated: no persistent history; persistent: use config paths",
    )
    args = parser.parse_args()

    cfg = load_model_config(Path(args.config)) if args.config else DEFAULT_CONFIG

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

    traces: list[dict[str, Any]] = []
    for i in range(int(args.runs)):
        res = run_engine(args.query, config=cfg, verbose=False)
        projection = generate_text(res.final_state.q, res.final_state.M, dim=cfg.embedding.dim) if res.success else None
        traces.append(engine_result_to_dict(res, projection=projection, include_atoms=True))
        traces[-1]["run_index"] = i + 1

    payload: dict[str, Any] = {
        "query": args.query,
        "mode": args.mode,
        "runs": int(args.runs),
        "config": model_config_to_dict(cfg),
        "trace": traces,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote trace: {out_path}")

    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

