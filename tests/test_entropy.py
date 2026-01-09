import unittest

from entropy_demo.embedding import embed_text
from entropy_demo.entropy import conflict_entropy, coverage_entropy, enrich_eta_from_v
from entropy_demo.types import MemoryAtom


def _atom(atom_id: str, v: str) -> MemoryAtom:
    m = MemoryAtom(
        id=atom_id,
        q_i="",
        v_i=v,
        z_i=embed_text(v, dim=64),
        c_i=0.1,
        s_i=0.5,
        eta_i={},
    )
    enrich_eta_from_v(m)
    return m


class TestEntropy(unittest.TestCase):
    def test_ecov_empty_is_one(self) -> None:
        z_q = embed_text("anything", dim=64)
        ecov, sim_max = coverage_entropy(z_q, [], clip_cosine=True)
        self.assertEqual(ecov, 1.0)
        self.assertEqual(sim_max, 0.0)

    def test_econf_known_conflict_pair(self) -> None:
        a = _atom("a", "FACT:x=y")
        b = _atom("b", "NOT:x=y")
        self.assertEqual(conflict_entropy([a, b], include_constraints=False), 1.0)

    def test_conflict_mediation_constraint(self) -> None:
        a = _atom("a", "FACT:x=y")
        b = _atom("b", "NOT:x=y")
        constraint = MemoryAtom(
            id="c1",
            q_i="",
            v_i="CONSTRAINT: reconcile",
            z_i=embed_text("constraint", dim=64),
            c_i=0.1,
            s_i=0.1,
            eta_i={"reconcile_pairs": [["a", "b"]]},
        )
        self.assertEqual(conflict_entropy([a, b, constraint], include_constraints=False), 0.0)

    def test_econf_conflict_from_metadata(self) -> None:
        a = _atom("a", "NOTE: something")
        b = _atom("b", "NOTE: something else")
        a.eta_i.update({"claim_key": "k", "polarity": 1})
        b.eta_i.update({"claim_key": "k", "polarity": -1})
        self.assertEqual(conflict_entropy([a, b], include_constraints=False), 1.0)
