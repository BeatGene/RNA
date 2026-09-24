import tempfile
import unittest
from pathlib import Path

from length_sampling import long_rna_weights


class LongRnaWeightsTest(unittest.TestCase):
    def test_boundary_and_pdb_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "selection_audit.tsv"
            table.write_text(
                "PDB_ID\tRNA_LENGTH\n1ABC\t50\n2DEF\t51\n",
                encoding="utf-8",
            )
            train = root / "train"
            files = [train / "1abc/seed_300/sample_0.pt",
                     train / "2def/seed_300/sample_0.pt",
                     train / "2def/seed_300/sample_1.pt"]
            weights, counts = long_rna_weights(files, train, table, factor=2)
            self.assertEqual(weights, [1, 2, 2])
            self.assertEqual(counts, {"long_pt": 2, "short_pt": 1,
                                      "long_pdb": 1, "short_pdb": 1})


if __name__ == "__main__":
    unittest.main()
