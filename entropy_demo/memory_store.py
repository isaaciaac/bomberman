"""Long-term memory store (calM) loading, persistence, and capacity control."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import CapacityConfig, EmbeddingConfig, WriteBackConfig
from .embedding import cos, embed_average, embed_text
from .entropy import enrich_eta_from_v
from .types import MemoryAtom, ensure_eta_dict, is_constraint_atom


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_fingerprint(atom: MemoryAtom) -> str:
    """Hash fields that define content identity for de-duplication/write-back."""

    eta = dict(atom.eta_i)
    # Remove run-specific keys if present.
    eta.pop("rewrite_type", None)
    # Canonicalize reconcile_pairs ordering.
    pairs = eta.get("reconcile_pairs")
    if isinstance(pairs, list):
        normalized: list[list[str]] = []
        for p in pairs:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                a, b = str(p[0]), str(p[1])
                normalized.append(sorted([a, b]))
        normalized.sort()
        eta["reconcile_pairs"] = normalized

    subsumes = eta.get("subsumes")
    if isinstance(subsumes, list):
        eta["subsumes"] = sorted({str(x) for x in subsumes})

    payload = {
        "q_i": atom.q_i,
        "v_i": atom.v_i,
        "c_i": float(atom.c_i),
        "s_i": float(atom.s_i),
        "eta_i": eta,
    }
    return hashlib.md5(_stable_json(payload).encode("utf-8")).hexdigest()


def _writeback_id(atom: MemoryAtom) -> str:
    kind = str(atom.eta_i.get("rewrite_type", "mem"))
    fp = _content_fingerprint(atom)[:10]
    return f"wb_{kind}_{fp}"


def load_memory_file(path: Path, *, embedding: EmbeddingConfig) -> list[MemoryAtom]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON list")
    out: list[MemoryAtom] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        v_i = str(item.get("v_i", ""))
        atom = MemoryAtom(
            id=str(item.get("id", "")),
            q_i=str(item.get("q_i", "")),
            v_i=v_i,
            z_i=embed_text(
                v_i,
                dim=embedding.dim,
                token_min_len=embedding.token_min_len,
                use_english_stopwords=embedding.use_english_stopwords,
            ),
            c_i=float(item.get("c_i", 0.0)),
            s_i=float(item.get("s_i", 0.0)),
            eta_i=ensure_eta_dict(item.get("eta_i")),
        )
        if not atom.id:
            atom.id = _writeback_id(atom)
        enrich_eta_from_v(atom)
        out.append(atom)
    return out


def save_memory_file(path: Path, atoms: Iterable[MemoryAtom]) -> None:
    records: list[dict[str, Any]] = []
    for a in atoms:
        records.append(
            {
                "id": a.id,
                "q_i": a.q_i,
                "v_i": a.v_i,
                "c_i": float(a.c_i),
                "s_i": float(a.s_i),
                "eta_i": dict(a.eta_i),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class MemoryStore:
    """Long-term memory store calM.

    - Seed memories are loaded from `seed_path` each run.
    - Optional write-back memories live in `writeback.persist_path` and can be
      enabled via WriteBackConfig.
    """

    seed_path: Path
    embedding: EmbeddingConfig
    writeback: WriteBackConfig

    def load(self) -> None:
        self._seed = load_memory_file(self.seed_path, embedding=self.embedding)
        self._writeback_atoms = []
        if self.writeback.enabled and self.writeback.persist_path.exists():
            self._writeback_atoms = load_memory_file(self.writeback.persist_path, embedding=self.embedding)

        # Stable de-dup by id, allowing write-back atoms to override seed atoms.
        #
        # This makes "overlay" records possible: we can persist updates to an
        # existing atom (e.g. suppress its valence after folding) without editing
        # the seed file.
        idx: dict[str, int] = {}
        merged: list[MemoryAtom] = []
        for a in self._seed:
            idx[a.id] = len(merged)
            merged.append(a)
        for a in self._writeback_atoms:
            if a.id in idx:
                merged[idx[a.id]] = a
            else:
                idx[a.id] = len(merged)
                merged.append(a)
        self._all = merged

    @property
    def atoms(self) -> list[MemoryAtom]:
        return list(self._all)

    def get_by_id(self, atom_id: str) -> MemoryAtom | None:
        for a in self._all:
            if a.id == atom_id:
                return a
        return None

    def total_cost(self) -> float:
        return float(sum(a.c_i for a in self._all))

    def effective_cost(self) -> float:
        """Effective cost used for capacity control (demo approximation).

        Atoms listed in an abstraction's `eta_i.subsumes` are treated as inactive.
        """

        subsumed: set[str] = set()
        for a in self._all:
            subsumes = a.eta_i.get("subsumes", [])
            if isinstance(subsumes, list):
                subsumed.update(str(x) for x in subsumes)
        return float(sum(a.c_i for a in self._all if a.id not in subsumed))

    def capacity_cost(self, *, capacity: CapacityConfig) -> float:
        """Cost metric used for the capacity check."""

        if capacity.cost_mode == "total":
            return self.total_cost()
        return self.effective_cost()

    def _load_writeback_stats(self) -> dict[str, int]:
        if not self.writeback.stats_path.exists():
            return {}
        try:
            raw = json.loads(self.writeback.stats_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, int] = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, int):
                out[k] = v
        return out

    def _save_writeback_stats(self, stats: Mapping[str, int]) -> None:
        self.writeback.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.writeback.stats_path.write_text(json.dumps(dict(stats), indent=2, sort_keys=True), encoding="utf-8")

    def write_back(self, atoms: list[MemoryAtom], *, success: bool) -> list[MemoryAtom]:
        """Optionally persist a subset of atoms into long-term memory.

        This implements a minimal "repeated validation" rule:
        - After each *successful* run, a candidate atom's content fingerprint count increases.
        - When the count reaches `min_successes`, the atom is persisted (deduped by content id).
        """

        if not self.writeback.enabled or not success:
            return []

        allowed = set(self.writeback.include_kinds)
        candidates: list[MemoryAtom] = []
        for a in atoms:
            kind = str(a.eta_i.get("rewrite_type", ""))
            if kind in allowed:
                candidates.append(a)

        candidates = candidates[: self.writeback.max_atoms_per_run]

        stats = self._load_writeback_stats()
        persisted: list[MemoryAtom] = []

        existing_ids = {a.id for a in self._writeback_atoms}
        existing_fps = {_content_fingerprint(a) for a in self._writeback_atoms}

        for a in candidates:
            fp = _content_fingerprint(a)
            stats[fp] = int(stats.get(fp, 0)) + 1
            if stats[fp] < self.writeback.min_successes:
                continue
            if fp in existing_fps:
                continue

            # Persist a stable copy with a deterministic id.
            new_id = _writeback_id(a)
            if new_id in existing_ids:
                continue

            persisted_atom = MemoryAtom(
                id=new_id,
                q_i=a.q_i,
                v_i=a.v_i,
                z_i=a.z_i.copy(),
                c_i=float(a.c_i),
                s_i=float(a.s_i),
                eta_i=dict(a.eta_i),
            )
            persisted.append(persisted_atom)
            existing_ids.add(new_id)
            existing_fps.add(fp)

        if persisted:
            self._writeback_atoms.extend(persisted)
            save_memory_file(self.writeback.persist_path, self._writeback_atoms)
            self.load()

        self._save_writeback_stats(stats)
        return persisted

    def enforce_capacity(self, *, capacity: CapacityConfig) -> list[MemoryAtom]:
        """Apply an abstraction/compression step if the long-term store is over budget."""

        if not capacity.enabled or not self.writeback.enabled:
            return []
        if self.capacity_cost(capacity=capacity) <= capacity.c_max:
            return []

        created: list[MemoryAtom] = []
        for _ in range(capacity.max_folds_per_write):
            if self.capacity_cost(capacity=capacity) <= capacity.c_max:
                break
            folded = self._fold_once(capacity=capacity)
            if folded is None:
                break
            created.append(folded)

        if created:
            save_memory_file(self.writeback.persist_path, self._writeback_atoms)
            self.load()
        return created

    def _suppress_atom(
        self,
        atom_id: str,
        *,
        subsumed_by: str,
        suppressed_valence: float,
        allow_seed_overlays: bool,
    ) -> None:
        for a in self._writeback_atoms:
            if a.id != atom_id:
                continue
            a.s_i = float(suppressed_valence)
            a.eta_i["subsumed_by"] = subsumed_by
            return

        if not allow_seed_overlays:
            return
        base = self.get_by_id(atom_id)
        if base is None:
            return

        overlay = MemoryAtom(
            id=base.id,
            q_i=base.q_i,
            v_i=base.v_i,
            z_i=base.z_i.copy(),
            c_i=float(base.c_i),
            s_i=float(suppressed_valence),
            eta_i=dict(base.eta_i),
        )
        overlay.eta_i["subsumed_by"] = subsumed_by
        overlay.eta_i["overlay"] = True
        self._writeback_atoms.append(overlay)

    def _fold_once(self, *, capacity: CapacityConfig) -> MemoryAtom | None:
        pool = [a for a in self._all if not is_constraint_atom(a)]
        if len(pool) < 2:
            return None

        subsumed: set[str] = set()
        for a in self._all:
            subsumes = a.eta_i.get("subsumes", [])
            if isinstance(subsumes, list):
                subsumed.update(str(x) for x in subsumes)

        candidates = [a for a in pool if a.id not in subsumed]
        if len(candidates) < 2:
            return None

        if capacity.fold_policy == "lowest_valence":
            candidates.sort(key=lambda a: float(a.s_i))
            a, b = candidates[0], candidates[1]
        else:
            # most_similar_pair
            best_pair: tuple[MemoryAtom, MemoryAtom] | None = None
            best_sim = -1.0
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    sim = cos(candidates[i].z_i, candidates[j].z_i)
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (candidates[i], candidates[j])
            if best_pair is None:
                return None
            a, b = best_pair

        z_abs = embed_average([a.z_i, b.z_i])
        sum_cost = float(a.c_i + b.c_i)
        new_atom = MemoryAtom(
            id="",  # assigned below
            q_i="folding",
            v_i=f"ABSTRACT: fold {a.id}+{b.id} (capacity)",
            z_i=z_abs,
            c_i=min(capacity.fold_cost_ratio * sum_cost, sum_cost),
            s_i=capacity.fold_valence,
            eta_i={"rewrite_type": "fold", "subsumes": [a.id, b.id]},
        )
        new_atom.id = _writeback_id(new_atom)
        if any(x.id == new_atom.id for x in self._writeback_atoms):
            return None
        self._writeback_atoms.append(new_atom)

        self._suppress_atom(
            a.id,
            subsumed_by=new_atom.id,
            suppressed_valence=capacity.suppressed_valence,
            allow_seed_overlays=capacity.allow_seed_overlays,
        )
        self._suppress_atom(
            b.id,
            subsumed_by=new_atom.id,
            suppressed_valence=capacity.suppressed_valence,
            allow_seed_overlays=capacity.allow_seed_overlays,
        )
        return new_atom
