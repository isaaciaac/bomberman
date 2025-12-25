"""Expression/projection layer: a trivial G(q, M) -> text."""

from __future__ import annotations

from typing import Sequence

from .embedding import cos, embed_text
from .types import MemoryAtom, is_constraint_atom


def generate_text(q: str, M: Sequence[MemoryAtom], *, dim: int = 256) -> str:
    """Produce a short text summary using selected memories."""

    z_q = embed_text(q, dim=dim)
    ranked = sorted(
        (m for m in M if not is_constraint_atom(m)),
        key=lambda m: cos(z_q, m.z_i) * max(0.0, float(m.s_i)),
        reverse=True,
    )
    top = ranked[:5]

    lines: list[str] = []
    lines.append("OUTPUT")
    lines.append(f"q: {q}")
    if not top:
        lines.append("No usable memories selected.")
        return "\n".join(lines)

    lines.append("Selected memory atoms:")
    for m in top:
        lines.append(f"- {m.id}: {m.v_i}")
    return "\n".join(lines)
