import tempfile
import unittest
from pathlib import Path

from dataclasses import replace

from entropy_demo.config import DEFAULT_CONFIG
from entropy_demo.engine import run_engine


class TestEngine(unittest.TestCase):
    def test_s2_strict_descent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            history_path = Path(td) / "history.json"
            cfg = replace(
                DEFAULT_CONFIG,
                stability=replace(DEFAULT_CONFIG.stability, history_path=history_path),
                entropy=replace(DEFAULT_CONFIG.entropy, epsilon=0.05),
                rewrite=replace(DEFAULT_CONFIG.rewrite, tmax=4),
                retrieval=replace(DEFAULT_CONFIG.retrieval, k=3),
            )

            # Use a query unlikely to match the seed so E_cov starts high and BRIDGE triggers.
            res = run_engine(
                "completely_unrelated_zzzz_12345",
                config=cfg,
                verbose=False,
            )
            entropies = [step.entropy.total for step in res.steps]
            # Ensure at least one rewrite happened.
            self.assertGreaterEqual(len(entropies), 1)
            # Strict descent across accepted steps: E_t < E_{t-1}
            for i in range(1, len(entropies)):
                self.assertLess(entropies[i], entropies[i - 1])
