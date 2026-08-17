from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_rnafm_embeddings.py"
SPEC = importlib.util.spec_from_file_location(
    "generate_rnafm_embeddings", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class GenerateRnaFmEmbeddingsTests(unittest.TestCase):
    def test_sliding_windows_cover_every_residue(self) -> None:
        cases = {
            1022: [0],
            1023: [0, 1],
            1533: [0, 511],
            2880: [0, 766, 1532, 1858],
            3774: [0, 766, 1532, 2298, 2752],
        }
        for length, expected_starts in cases.items():
            with self.subTest(length=length):
                starts = GENERATOR.sliding_window_starts(length, 1022, 256)
                self.assertEqual(starts, expected_starts)
                coverage = [0] * length
                for start in starts:
                    for index in range(start, min(start + 1022, length)):
                        coverage[index] += 1
                self.assertGreaterEqual(min(coverage), 1)

    def test_make_batches_respects_padded_token_budget(self) -> None:
        items = [
            ("a", "A" * 100),
            ("b", "A" * 200),
            ("c", "A" * 300),
        ]
        batches = GENERATOR.make_batches(
            items,
            special_tokens=2,
            max_batch_tokens=500,
            max_batch_size=64,
        )
        self.assertEqual([[key for key, _ in batch] for batch in batches], [["a", "b"], ["c"]])
        for batch in batches:
            padded_tokens = max(len(sequence) + 2 for _, sequence in batch) * len(batch)
            self.assertLessEqual(padded_tokens, 500)


if __name__ == "__main__":
    unittest.main()
