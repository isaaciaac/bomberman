import tempfile
import unittest
from pathlib import Path

from entropy_demo.engine import EngineConfig, run_engine


class TestEngine(unittest.TestCase):
    def test_s2_strict_descent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            history_path = Path(td) / "history.json"
            cfg = EngineConfig(
                k=3,
                epsilon=0.3,
                tmax=4,
                history_path=history_path,
            )
            # Use a query unlikely to match the seed so E_cov starts high and BRIDGE triggers.
            res = run_engine(
                "unseen_query_tokens_12345",
                seed_path=Path("data/memory_seed.json"),
                config=cfg,
                verbose=False,
            )
            entropies = [step.entropy.total for step in res.steps]
            # Ensure at least one rewrite happened.
            self.assertGreaterEqual(len(entropies), 1)
            # Strict descent across accepted steps: E_t < E_{t-1}
            for i in range(1, len(entropies)):
                self.assertLess(entropies[i], entropies[i - 1])

