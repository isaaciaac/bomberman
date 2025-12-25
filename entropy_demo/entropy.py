"""Expressibility entropy gate E(q, M) and its components."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embedding import cos, tokenize
from .types import MemoryAtom, is_constraint_atom, memory_signature

_KV_RE = re.compile(r"^(FACT|NOT)\s*:\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$")


def enrich_eta_from_v(atom: MemoryAtom) -> None:
    """Parse simple FACT/NOT patterns and store parsed results in `eta_i`."""

    m = _KV_RE.match(atom.v_i)
    if not m:
        return

    kind = m.group(1).upper()
    key = m.group(2)
    value = m.group(3)
    polarity = 1 if kind == "FACT" else -1

    atom.eta_i.setdefault("parsed", {})
    parsed: dict[str, Any] = atom.eta_i["parsed"]
    parsed.update({"kind": kind, "key": key, "value": value, "polarity": polarity})


def _reconciled_pair_ids(M: Sequence[MemoryAtom]) -> set[tuple[str, str]]:
    reconciled: set[tuple[str, str]] = set()
    for atom in M:
        if not is_constraint_atom(atom):
            continue
        pairs = atom.eta_i.get("reconcile_pairs", [])
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = str(pair[0]), str(pair[1])
            reconciled.add(tuple(sorted((a, b))))
    return reconciled


def _conflict_key_and_polarity(atom: MemoryAtom) -> tuple[str, int] | None:
    """Extract a (key, polarity) pair for conflict detection.

    Preference order:
    1) Parsed from v_i patterns into eta_i["parsed"].
    2) Explicit metadata fields in eta_i, e.g. claim_key/key + polarity.
    """

    parsed = atom.eta_i.get("parsed") or {}
    key = parsed.get("key")
    polarity = parsed.get("polarity")
    if isinstance(key, str) and isinstance(polarity, int) and polarity in (-1, 1):
        return key, polarity

    eta = atom.eta_i
    meta_key = eta.get("claim_key", eta.get("key"))
    meta_pol = eta.get("polarity")
    if isinstance(meta_key, str) and isinstance(meta_pol, int) and meta_pol in (-1, 1):
        return meta_key, meta_pol
    return None


def chi(mi: MemoryAtom, mj: MemoryAtom, *, M: Sequence[MemoryAtom]) -> int:
    """Strong conflict indicator chi(mi, mj) in {0,1} (demo heuristic)."""

    if is_constraint_atom(mi) or is_constraint_atom(mj):
        return 0

    pair = tuple(sorted((mi.id, mj.id)))
    if pair in _reconciled_pair_ids(M):
        return 0

    ki = _conflict_key_and_polarity(mi)
    kj = _conflict_key_and_polarity(mj)
    if ki is None or kj is None:
        return 0

    key_i, pol_i = ki
    key_j, pol_j = kj
    if key_i == key_j and pol_i == -pol_j:
        return 1
    return 0


def coverage_entropy(z_q: np.ndarray, M: Sequence[MemoryAtom]) -> float:
    """E_cov(q,M) = 1 - sim_max(q,M)."""

    if not M:
        return 1.0
    sim_max = max(cos(z_q, m.z_i) for m in M)
    return float(1.0 - sim_max)


def conflict_entropy(M: Sequence[MemoryAtom]) -> float:
    """E_conf(M) = average strong-conflict rate over pairs."""

    n = len(M)
    if n < 2:
        return 0.0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += chi(M[i], M[j], M=M)
    return float((2.0 / (n * (n - 1))) * total)


def cluster_query(q: str, *, n_tokens: int = 4) -> str:
    toks = tokenize(q)
    head = " ".join(toks[:n_tokens])
    import hashlib

    return hashlib.md5(head.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HistoryConfig:
    p0: float = 0.5
    ema_rate: float = 0.2
    include_constraints_in_sig: bool = False
    include_generated_in_sig: bool = False


class HistoryStore:
    """Persistent mapping H(cluster(q), sig(M)) -> p_succ in [0,1]."""

    def __init__(self, path: Path, *, config: HistoryConfig):
        self._path = path
        self._config = config
        self._data: dict[str, dict[str, float]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get_p_succ(self, *, cluster: str, sig: str) -> float:
        self._ensure_loaded()
        return float(self._data.get(cluster, {}).get(sig, self._config.p0))

    def update(self, *, cluster: str, sig: str, success: bool) -> float:
        self._ensure_loaded()
        old = float(self._data.get(cluster, {}).get(sig, self._config.p0))
        target = 1.0 if success else 0.0
        new = (1.0 - self._config.ema_rate) * old + self._config.ema_rate * target
        self._data.setdefault(cluster, {})[sig] = float(new)
        self.save()
        return float(new)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    @property
    def config(self) -> HistoryConfig:
        return self._config


@dataclass(frozen=True)
class EntropyWeights:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2

    def __post_init__(self) -> None:
        s = self.alpha + self.beta + self.gamma
        if not np.isclose(s, 1.0):
            raise ValueError(f"alpha+beta+gamma must sum to 1 (got {s})")


@dataclass(frozen=True)
class EntropyBreakdown:
    total: float
    E_cov: float
    E_conf: float
    E_stab: float
    sim_max: float
    p_succ: float
    cluster: str
    sig: str


def compute_entropy(
    q: str,
    *,
    z_q: np.ndarray,
    M: Sequence[MemoryAtom],
    history: HistoryStore,
    weights: EntropyWeights,
) -> EntropyBreakdown:
    """Compute total E(q,M)=alpha*E_cov+beta*E_conf+gamma*E_stab and return components."""

    E_cov = coverage_entropy(z_q, M)
    sim_max = 0.0 if not M else float(max(cos(z_q, m.z_i) for m in M))
    E_conf = conflict_entropy(M)

    cluster = cluster_query(q)
    sig = memory_signature(
        M,
        include_constraints=history.config.include_constraints_in_sig,
        include_generated=history.config.include_generated_in_sig,
    )
    p_succ = history.get_p_succ(cluster=cluster, sig=sig)
    E_stab = float(1.0 - p_succ)

    total = weights.alpha * E_cov + weights.beta * E_conf + weights.gamma * E_stab
    return EntropyBreakdown(
        total=float(total),
        E_cov=float(E_cov),
        E_conf=float(E_conf),
        E_stab=float(E_stab),
        sim_max=float(sim_max),
        p_succ=float(p_succ),
        cluster=cluster,
        sig=sig,
    )
