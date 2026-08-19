import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_chain_maps import CandidatePath, ChainRecord
from resolve_chain_maps_by_coordinates import (
    CandidateMapping,
    CoordinateChain,
    STATUS_PASS,
    choose_consensus_mappings,
    kabsch_transform,
    is_blocking_prediction_input_issue,
    rows_for_variant,
    solve_candidate_mapping,
    ResolvedTarget,
    write_resolved_target,
)


def coordinate_chain(
    label: str, auth: str, sequence: str, points: list[list[float]]
) -> CoordinateChain:
    return CoordinateChain(
        ChainRecord(label, auth, sequence),
        {
            index: np.asarray(point, dtype=np.float64)
            for index, point in enumerate(points, start=1)
        },
    )


class ResolveChainMapsByCoordinatesTests(unittest.TestCase):
    def test_prediction_sequence_decode_difference_is_nonblocking(self):
        self.assertFalse(
            is_blocking_prediction_input_issue(
                "predicted CIF theoretical sequence differs from JSON input for chain A"
            )
        )
        self.assertTrue(
            is_blocking_prediction_input_issue(
                "predicted chain A is absent from base chain_map.tsv"
            )
        )
        self.assertTrue(
            is_blocking_prediction_input_issue(
                "predicted CIF theoretical-sequence length differs from JSON input "
                "for chain A: CIF=3, JSON=4"
            )
        )
        self.assertTrue(
            is_blocking_prediction_input_issue(
                "predicted CIF theoretical sequence conflicts with JSON input for chain A"
            )
        )

    def test_kabsch_recovers_rigid_transform(self):
        mobile = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        reference = mobile @ rotation + np.asarray([5.0, -2.0, 7.0])
        fitted_rotation, translation = kabsch_transform(mobile, reference)
        np.testing.assert_allclose(
            mobile @ fitted_rotation + translation, reference, atol=1.0e-10
        )

    def test_complex_alignment_resolves_identical_chain_sequences(self):
        local = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        gt_x = local
        gt_y = local + np.asarray([9.0, 3.0, 1.0])
        # Prediction labels are intentionally reversed relative to GT labels.
        pred_a = gt_y + np.asarray([100.0, -20.0, 4.0])
        pred_b = gt_x + np.asarray([100.0, -20.0, 4.0])
        pred = [
            coordinate_chain("A", "A", "ACGU", pred_a.tolist()),
            coordinate_chain("B", "B", "ACGU", pred_b.tolist()),
        ]
        gt = [
            coordinate_chain("X", "GX", "ACGU", gt_x.tolist()),
            coordinate_chain("Y", "GY", "ACGU", gt_y.tolist()),
        ]
        options, exact, issues = solve_candidate_mapping(
            pred, gt, max_iterations=20, tie_tolerance=1.0e-6
        )
        self.assertTrue(exact)
        self.assertEqual(issues, [])
        self.assertEqual(len(options), 1)
        mapping = next(iter(options))
        self.assertEqual(mapping, (("A", "Y"), ("B", "X")))
        self.assertAlmostEqual(options[mapping], 0.0, places=8)

    def test_majority_breaks_one_candidates_equal_rmsd_tie(self):
        map_xy = (("A", "X"), ("B", "Y"))
        map_yx = (("A", "Y"), ("B", "X"))
        candidates = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sample in range(3):
                candidates.append(
                    CandidateMapping(
                        candidate=CandidatePath(42, sample, root / f"{sample}.cif"),
                        pred_chains=[],
                        option_scores={map_xy: 1.0},
                    )
                )
            candidates.append(
                CandidateMapping(
                    candidate=CandidatePath(66, 0, root / "tie.cif"),
                    pred_chains=[],
                    option_scores={map_xy: 1.0, map_yx: 1.0},
                )
            )
            choose_consensus_mappings(candidates)
        self.assertEqual(candidates[-1].chosen_mapping, map_xy)
        self.assertTrue(candidates[-1].consensus_tie_break)

    def test_exact_false_row_can_never_be_pass(self):
        pred = [coordinate_chain("A", "A", "ACGU", [[0, 0, 0]] * 4)]
        gt = [coordinate_chain("X", "GX", "ACGA", [[0, 0, 0]] * 4)]
        candidate = CandidateMapping(
            CandidatePath(42, 0, Path("pred.cif")),
            pred,
            {(('A', 'X'),): 0.0},
        )
        rows = rows_for_variant((("A", "X"),), candidate, gt, STATUS_PASS)
        self.assertEqual(rows[0]["exact_match"], "False")
        self.assertNotEqual(rows[0]["status"], STATUS_PASS)

    def test_writer_coalesces_candidates_with_same_mapping(self):
        mapping = (("A", "X"), ("B", "Y"))
        pred = [
            coordinate_chain("A", "A", "ACGU", [[0, 0, 0]] * 4),
            coordinate_chain("B", "B", "ACGU", [[1, 0, 0]] * 4),
        ]
        gt = [
            coordinate_chain("X", "GX", "ACGU", [[0, 0, 0]] * 4),
            coordinate_chain("Y", "GY", "ACGU", [[1, 0, 0]] * 4),
        ]
        candidates = [
            CandidateMapping(
                CandidatePath(42, sample, Path(f"pred_{sample}.cif")),
                pred,
                {mapping: 0.1},
                chosen_mapping=mapping,
                chosen_rmsd=0.1,
            )
            for sample in range(2)
        ]
        result = ResolvedTarget(
            pdb_id="1ABC",
            gt_chains=gt,
            inventory_count=2,
            candidates=candidates,
            inventory_issues=[],
            fatal_issues=[],
            review_reasons=[],
            variants={mapping: candidates},
            status=STATUS_PASS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            count = write_resolved_target(
                Path(tmp),
                result,
                representative_atom="C4'",
                min_gt_coverage=0.5,
                min_dominant_fraction=0.2,
                overwrite=False,
            )
            self.assertEqual(count, 1)
            target = Path(tmp) / "1ABC"
            self.assertTrue(
                (target / "chain_map_variants" / "mapping_01" / "chain_map.tsv").is_file()
            )
            manifest = (target / "chain_map.txt").read_text(encoding="utf-8")
            self.assertIn("seed_42_sample_0, seed_42_sample_1", manifest)
            self.assertIn("mapping_variants: 1", manifest)


if __name__ == "__main__":
    unittest.main()
