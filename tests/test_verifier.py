import unittest

from entropy_demo.config import VerifierConfig
from entropy_demo.embedding import embed_text
from entropy_demo.entropy import enrich_eta_from_v
from entropy_demo.types import MemoryAtom
from entropy_demo.verifier import verify_state


def _mk(atom_id: str, *, v_i: str, eta: dict | None = None) -> MemoryAtom:
    m = MemoryAtom(
        id=atom_id,
        q_i="",
        v_i=v_i,
        z_i=embed_text(v_i, dim=64),
        c_i=0.1,
        s_i=0.5,
        eta_i=dict(eta or {}),
    )
    enrich_eta_from_v(m)
    return m


class TestVerifier(unittest.TestCase):
    def test_requires_no_unreconciled_conflicts(self) -> None:
        cfg = VerifierConfig(
            enabled=True,
            require_no_unreconciled_conflicts=True,
            min_non_generated_atoms=1,
            require_claim_keys_from_query=False,
        )
        a = _mk("a", v_i="FACT:x=y")
        b = _mk("b", v_i="NOT:x=y")
        res = verify_state("q", [a, b], cfg=cfg)
        self.assertFalse(res.passed)

        constraint = _mk(
            "c",
            v_i="CONSTRAINT: reconcile",
            eta={"reconcile_pairs": [["a", "b"]]},
        )
        res2 = verify_state("q", [a, b, constraint], cfg=cfg)
        self.assertTrue(res2.passed)

    def test_query_required_claim_keys(self) -> None:
        cfg = VerifierConfig(
            enabled=True,
            require_no_unreconciled_conflicts=False,
            min_non_generated_atoms=1,
            require_claim_keys_from_query=True,
        )

        q = "test REQ[answer_in_state]"
        good = _mk("m", v_i="claim", eta={"claim_key": "answer_in_state", "polarity": 1})
        self.assertTrue(verify_state(q, [good], cfg=cfg).passed)

        bad = _mk("m2", v_i="claim", eta={"claim_key": "other_key", "polarity": 1})
        self.assertFalse(verify_state(q, [bad], cfg=cfg).passed)

    def test_config_required_claim_keys(self) -> None:
        cfg = VerifierConfig(
            enabled=True,
            require_no_unreconciled_conflicts=False,
            min_non_generated_atoms=1,
            require_claim_keys_from_query=False,
            required_claim_keys=("k1", "k2"),
        )

        a = _mk("a", v_i="claim", eta={"claim_key": "k1", "polarity": 1})
        b = _mk("b", v_i="claim", eta={"claim_key": "k2", "polarity": 1})
        self.assertTrue(verify_state("q", [a, b], cfg=cfg).passed)

        self.assertFalse(verify_state("q", [a], cfg=cfg).passed)

    def test_min_non_generated_atoms(self) -> None:
        cfg = VerifierConfig(
            enabled=True,
            require_no_unreconciled_conflicts=False,
            min_non_generated_atoms=1,
            min_evidence_sim=0.0,
            evidence_clip_cosine=True,
            require_claim_keys_from_query=False,
        )

        bridge = _mk("rw_bridge_1", v_i="BRIDGE: restate query -> q", eta={"rewrite_type": "bridge"})
        self.assertFalse(verify_state("q", [bridge], cfg=cfg).passed)

    def test_min_evidence_similarity(self) -> None:
        cfg = VerifierConfig(
            enabled=True,
            require_no_unreconciled_conflicts=False,
            min_non_generated_atoms=1,
            min_evidence_sim=0.1,
            evidence_clip_cosine=True,
            min_evidence_token_overlap=0,
            evidence_token_min_len=3,
            evidence_use_english_stopwords=True,
            require_claim_keys_from_query=False,
        )

        # Use a non-generated atom with a zero vector to make similarity 0 deterministically.
        z = embed_text("anything", dim=64)
        z[:] = 0.0
        atom = MemoryAtom(
            id="m",
            q_i="",
            v_i="note",
            z_i=z,
            c_i=0.1,
            s_i=0.5,
            eta_i={},
        )
        res = verify_state("hello", [atom], cfg=cfg)
        self.assertFalse(res.passed)

    def test_min_evidence_token_overlap(self) -> None:
        cfg = VerifierConfig(
            enabled=True,
            require_no_unreconciled_conflicts=False,
            min_non_generated_atoms=1,
            min_evidence_sim=0.0,
            evidence_clip_cosine=True,
            min_evidence_token_overlap=1,
            evidence_token_min_len=3,
            evidence_use_english_stopwords=True,
            require_claim_keys_from_query=False,
        )

        atom = _mk("m1", v_i="Reasoning is structure first", eta={})
        self.assertTrue(verify_state("Reasoning and structure", [atom], cfg=cfg).passed)
        self.assertFalse(verify_state("capital of france", [atom], cfg=cfg).passed)
