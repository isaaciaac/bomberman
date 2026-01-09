import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from entropy_demo.config import DEFAULT_CONFIG
from entropy_demo.embedding import embed_text
from entropy_demo.memory_store import MemoryStore
from entropy_demo.types import MemoryAtom


class TestMemoryStore(unittest.TestCase):
    def test_writeback_requires_repeated_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            wb = replace(
                DEFAULT_CONFIG.writeback,
                enabled=True,
                persist_path=td_path / "memory_store.json",
                stats_path=td_path / "writeback_stats.json",
                min_successes=2,
                include_kinds=("bridge",),
            )
            store = MemoryStore(seed_path=DEFAULT_CONFIG.seed_path, embedding=DEFAULT_CONFIG.embedding, writeback=wb)
            store.load()

            v_i = "BRIDGE: restate query -> test"
            bridge = MemoryAtom(
                id="rw_bridge_1",
                q_i="test",
                v_i=v_i,
                z_i=embed_text(v_i, dim=DEFAULT_CONFIG.embedding.dim),
                c_i=0.1,
                s_i=0.6,
                eta_i={"rewrite_type": "bridge"},
            )

            persisted1 = store.write_back([bridge], success=True)
            self.assertEqual(persisted1, [])

            persisted2 = store.write_back([bridge], success=True)
            self.assertEqual(len(persisted2), 1)
            self.assertTrue(wb.persist_path.exists())
            store.load()
            self.assertIsNotNone(store.get_by_id(persisted2[0].id))

    def test_capacity_folding_suppresses_subsumed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            wb = replace(
                DEFAULT_CONFIG.writeback,
                enabled=True,
                persist_path=td_path / "memory_store.json",
                stats_path=td_path / "writeback_stats.json",
            )
            store = MemoryStore(seed_path=DEFAULT_CONFIG.seed_path, embedding=DEFAULT_CONFIG.embedding, writeback=wb)
            store.load()

            initial_cost = store.total_cost()
            cap = replace(
                DEFAULT_CONFIG.capacity,
                enabled=True,
                cost_mode="effective",
                c_max=initial_cost - 0.2,
                max_folds_per_write=5,
            )

            created = store.enforce_capacity(capacity=cap)
            self.assertGreaterEqual(len(created), 1)
            self.assertLessEqual(store.capacity_cost(capacity=cap), cap.c_max + 1e-6)

            fold = created[0]
            subsumes = fold.eta_i.get("subsumes", [])
            self.assertIsInstance(subsumes, list)
            self.assertGreaterEqual(len(subsumes), 2)
            for atom_id in subsumes[:2]:
                a = store.get_by_id(str(atom_id))
                self.assertIsNotNone(a)
                assert a is not None
                self.assertAlmostEqual(float(a.s_i), float(cap.suppressed_valence))
                self.assertEqual(a.eta_i.get("subsumed_by"), fold.id)

