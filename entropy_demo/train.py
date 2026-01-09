"""Virtual-environment training CLI: learns rewrite-type ordering templates."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG, load_model_config
from .virtual_env import VirtualEnvTrainConfig, train_rewrite_templates


def _load_queries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    queries: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        queries.append(s)
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description="Train rewrite templates (virtual environment)")
    parser.add_argument("--queries", required=True, help="Path to a text file with one query per line")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config path")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes")
    parser.add_argument("--explore", type=float, default=0.2, help="Exploration probability")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", type=str, default=".run/rewrite_templates.json", help="Output template JSON path")
    args = parser.parse_args()

    cfg = load_model_config(Path(args.config)) if args.config else DEFAULT_CONFIG
    queries = _load_queries(Path(args.queries))
    if not queries:
        raise SystemExit("no queries found")

    env = VirtualEnvTrainConfig(
        episodes=int(args.episodes),
        exploration_prob=float(args.explore),
        seed=int(args.seed),
        output_path=Path(args.out),
    )

    # Training is intended to be read-only w.r.t. the main persistent state.
    cfg = replace(cfg, writeback=replace(cfg.writeback, enabled=False), capacity=replace(cfg.capacity, enabled=False))

    templates = train_rewrite_templates(queries, model=cfg, env=env)
    print(f"wrote {len(templates)} bucket templates to {env.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

