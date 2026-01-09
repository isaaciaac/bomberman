"""CLI entry point: `python -m entropy_demo.cli --query \"...\"`."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG, load_model_config
from .engine import run_engine
from .generate import generate_text
from .reporting import engine_result_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Entropy-driven Memory Reconstruction demo")
    parser.add_argument("--query", required=True, help="Query string q")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config path")
    parser.add_argument("--k", type=int, default=None, help="S1 top-K (overrides config)")
    parser.add_argument("--epsilon", type=float, default=None, help="Expressibility threshold epsilon (overrides config)")
    parser.add_argument("--tmax", type=int, default=None, help="Max S2 steps T_max (overrides config)")
    parser.add_argument("--trace-json", type=str, default="", help="Optional path to write a JSON trace")
    args = parser.parse_args()

    cfg = load_model_config(Path(args.config)) if args.config else DEFAULT_CONFIG
    if args.k is not None:
        cfg = replace(cfg, retrieval=replace(cfg.retrieval, k=int(args.k)))
    if args.epsilon is not None:
        cfg = replace(cfg, entropy=replace(cfg.entropy, epsilon=float(args.epsilon)))
    if args.tmax is not None:
        cfg = replace(cfg, rewrite=replace(cfg.rewrite, tmax=int(args.tmax)))

    result = run_engine(args.query, config=cfg, verbose=True)

    projection: str | None = None
    print()
    print("Result")
    print(f"success={result.success} reason={result.reason}")
    fe = result.final_entropy
    print(
        "E={:.4f} (E_cov={:.4f} E_conf={:.4f} E_stab={:.4f} p_succ={:.2f})".format(
            fe.total, fe.E_cov, fe.E_conf, fe.E_stab, fe.p_succ
        )
    )
    if cfg.verifier.enabled:
        vf = result.verifier_final
        print(f"V: passed={vf.passed} reasons={list(vf.reasons)}")

    print()
    print("Retrieved M0 (S1):")
    if not result.initial_state.M:
        print("- (empty)")
    else:
        for m in result.initial_state.M:
            print(f"- {m.id} (s={m.s_i:+.2f} c={m.c_i:.2f}) {m.v_i}")

    initial_ids = {m.id for m in result.initial_state.M}
    added = [m for m in result.final_state.M if m.id not in initial_ids]

    print()
    print("Added DeltaM (S2):")
    if not added:
        print("- (none)")
    else:
        for m in added:
            kind = m.eta_i.get("rewrite_type", "rewrite")
            print(f"- {m.id} ({kind}) {m.v_i}")

    print()
    if result.success:
        print("Projection (Layer 2):")
        projection = generate_text(result.final_state.q, result.final_state.M, dim=cfg.embedding.dim)
        print(projection)
        if result.writeback_persisted:
            print()
            print(f"Write-back persisted: {[m.id for m in result.writeback_persisted]}")
        if result.capacity_created:
            print()
            print(f"Capacity abstractions created: {[m.id for m in result.capacity_created]}")
        if args.trace_json:
            out = engine_result_to_dict(result, projection=projection, include_atoms=True)
            trace_path = Path(args.trace_json)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    print(f"REFUSAL: {result.reason}")
    if args.trace_json:
        out = engine_result_to_dict(result, projection=None, include_atoms=True)
        trace_path = Path(args.trace_json)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
