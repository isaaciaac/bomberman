import unittest

import numpy as np

from entropy_demo.embedding import embed_text


class TestEmbedding(unittest.TestCase):
    def test_deterministic(self) -> None:
        a = embed_text("Hello world", dim=256)
        b = embed_text("Hello world", dim=256)
        self.assertTrue(np.allclose(a, b))

