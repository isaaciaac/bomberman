"""Utilities to serialize run results for engineering verification.

The intent is to make runs auditable and batchable: a single input produces a
structured trace of what was retrieved (S1), what was added (S2), and what was
persisted (write-back / capacity).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .engine import EngineResult, StepRecord
from .types import MemoryAtom
from .verifier import VerifierResult


def memory_atom_to_dict(atom: MemoryAtom) -> dict[str, Any]:
    kind = str(atom.eta_i.get("rewrite_type", "seed"))
    return {
        "id": str(atom.id),
        "q_i": str(atom.q_i),
        "v_i": str(atom.v_i),
        "c_i": float(atom.c_i),
        "s_i": float(atom.s_i),
        "kind": kind,
        "eta_i": dict(atom.eta_i),
    }


def entropy_to_dict(e: Any) -> dict[str, Any]:
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


def step_record_to_dict(step: StepRecord) -> dict[str, Any]:
    return {
        "t": int(step.t),
        "M_size": int(step.M_size),
        "rewrite_kind": str(step.rewrite_kind),
        "added_ids": list(step.added_ids),
        "verifier": {"passed": bool(step.verifier_passed), "reasons": list(step.verifier_reasons)},
        "entropy": entropy_to_dict(step.entropy),
    }


def verifier_to_dict(v: VerifierResult) -> dict[str, Any]:
    return {"passed": bool(v.passed), "reasons": list(v.reasons), "metrics": dict(v.metrics)}


def engine_result_to_dict(
    res: EngineResult,
    *,
    projection: str | None = None,
    include_atoms: bool = True,
) -> dict[str, Any]:
    initial_ids = {m.id for m in res.initial_state.M}
    added = [m for m in res.final_state.M if m.id not in initial_ids]

    payload: dict[str, Any] = {
        "query": str(res.final_state.q),
        "success": bool(res.success),
        "reason": str(res.reason),
        "entropy": {
            "initial": entropy_to_dict(res.initial_entropy),
            "final": entropy_to_dict(res.final_entropy),
        },
        "verifier": {
            "initial": verifier_to_dict(res.verifier_initial),
            "final": verifier_to_dict(res.verifier_final),
        },
        "s1": {
            "retrieved_ids": list(res.initial_state.ids),
        },
        "s2": {
            "invoked": bool(len(res.steps) > 0),
            "steps": [step_record_to_dict(s) for s in res.steps],
            "added_ids": [m.id for m in added],
        },
        "writeback": {
            "persisted_ids": [m.id for m in res.writeback_persisted],
        },
        "capacity": {
            "created_ids": [m.id for m in res.capacity_created],
        },
    }

    if projection is not None:
        payload["projection"] = projection

    if include_atoms:
        payload["s1"]["retrieved_atoms"] = [memory_atom_to_dict(m) for m in res.initial_state.M]
        payload["s2"]["added_atoms"] = [memory_atom_to_dict(m) for m in added]

    return payload


def merge_metadata(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        out[str(k)] = v
    return out


def atoms_to_index(atoms: Iterable[MemoryAtom]) -> dict[str, dict[str, Any]]:
    return {a.id: memory_atom_to_dict(a) for a in atoms}
