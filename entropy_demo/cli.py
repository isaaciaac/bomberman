"""CLI entry point: `python -m entropy_demo.cli --query \"...\"`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import EngineConfig, run_engine
from .generate import generate_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Entropy-driven Memory Reconstruction demo")
    parser.add_argument("--query", required=True, help="Query string q")
    parser.add_argument("--k", type=int, default=5, help="S1 top-K")
    parser.add_argument("--epsilon", type=float, default=0.35, help="Expressibility threshold epsilon")
    parser.add_argument("--tmax", type=int, default=6, help="Max S2 steps T_max")
    args = parser.parse_args()

    seed_path = Path("data/memory_seed.json")
    cfg = EngineConfig(k=args.k, epsilon=args.epsilon, tmax=args.tmax)
    result = run_engine(args.query, seed_path=seed_path, config=cfg, verbose=True)

    print()
    print("Result")
    print(f"success={result.success} reason={result.reason}")
    fe = result.final_entropy
    print(
        "E={:.4f} (E_cov={:.4f} E_conf={:.4f} E_stab={:.4f} p_succ={:.2f})".format(
            fe.total, fe.E_cov, fe.E_conf, fe.E_stab, fe.p_succ
        )
    )

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
        print(generate_text(result.final_state.q, result.final_state.M, dim=cfg.dim))
        return 0
    print(f"REFUSAL: {result.reason}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
