import tempfile
import unittest
from pathlib import Path

from length_sampling import DistributedWeightedSampler, long_rna_weights


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

    def test_distributed_weighted_sampler_rank_epoch_and_weight(self):
        weights = [1, 1, 8] * 100
        samplers = [
            DistributedWeightedSampler(weights, num_replicas=2, rank=rank, seed=7)
            for rank in (0, 1)
        ]
        first = [list(sampler) for sampler in samplers]
        self.assertEqual([len(indices) for indices in first], [150, 150])
        self.assertNotEqual(first[0], first[1])
        self.assertEqual(first, [list(sampler) for sampler in samplers])
        self.assertGreater(sum(index % 3 == 2 for draws in first for index in draws), 210)
        for sampler in samplers:
            sampler.set_epoch(1)
        self.assertNotEqual(first, [list(sampler) for sampler in samplers])
        draws = list(DistributedWeightedSampler(weights, num_replicas=1, rank=0, seed=7))
        self.assertEqual(len(draws), 300)

    def test_distributed_weighted_sampler_rejects_invalid_weights(self):
        with self.assertRaises(ValueError):
            DistributedWeightedSampler([0, 1], num_replicas=1, rank=0)


if __name__ == "__main__":
    unittest.main()
