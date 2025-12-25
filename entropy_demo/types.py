"""Core types for the Entropy-driven Memory Reconstruction demo.

Memory atom fields (paper-aligned):
  (q_i, v_i, z_i, c_i, s_i, eta_i)

Engineering note:
  We also attach a stable string `id` for bookkeeping, signatures, and
  reconciliation pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

Eta = dict[str, Any]


@dataclass(slots=True)
class MemoryAtom:
    """A memory atom with the required fields."""

    id: str
    q_i: str
    v_i: str
    z_i: np.ndarray
    c_i: float
    s_i: float
    eta_i: Eta = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CognitiveState:
    """System state S = (q, M), where M is a finite candidate set.

    Critical: the state excludes any answer text.
    """

    q: str
    M: tuple[MemoryAtom, ...]

    def with_added(self, delta: Sequence[MemoryAtom]) -> "CognitiveState":
        """Monotone extension: return (q, M U DeltaM), de-duplicated by `id`."""

        if not delta:
            return self
        seen = {m.id for m in self.M}
        merged = list(self.M)
        for m in delta:
            if m.id not in seen:
                merged.append(m)
                seen.add(m.id)
        return CognitiveState(q=self.q, M=tuple(merged))

    @property
    def ids(self) -> list[str]:
        return [m.id for m in self.M]


def is_constraint_atom(atom: MemoryAtom) -> bool:
    return atom.v_i.startswith("CONSTRAINT:")


def is_bridge_atom(atom: MemoryAtom) -> bool:
    return atom.v_i.startswith("BRIDGE:")


def is_abstraction_atom(atom: MemoryAtom) -> bool:
    return atom.v_i.startswith("ABSTRACT:")


def memory_signature(
    memories: Iterable[MemoryAtom],
    *,
    include_constraints: bool,
    include_generated: bool,
) -> str:
    """Return a stable signature string based on sorted memory IDs.

    Deliberate choice for this demo:
    - Constraint atoms are excluded by default (they are mediation metadata and
      would fragment the stability table).
    - "Generated" atoms (BRIDGE/ABSTRACT) can be included/excluded via flag.
    """

    ids: list[str] = []
    for atom in memories:
        if not include_constraints and is_constraint_atom(atom):
            continue
        if not include_generated and (is_bridge_atom(atom) or is_abstraction_atom(atom)):
            continue
        ids.append(atom.id)
    ids.sort()
    return "|".join(ids)


def ensure_eta_dict(eta: Mapping[str, Any] | None) -> Eta:
    return dict(eta) if eta else {}
