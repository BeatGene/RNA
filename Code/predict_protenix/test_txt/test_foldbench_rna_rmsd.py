import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_foldbench_rna_rmsd import (
    Candidate,
    discover_candidates,
    load_valid_ost_output,
    read_manifest,
    write_manifest,
)
from rescue_foldbench_rna_rmsd import (
    audited_rescue_reason,
    has_audited_lddt_failure,
)
from summarize_foldbench_rna_rmsd import (
    describe_rmsd,
    load_candidate_results,
    read_ost_result,
    rna_input_size,
    run_summary,
    select_strict_rank1,
)
from build_foldbench_rna_rmsd_final_report import run as run_final_report
from build_foldbench_rna_multimetric_report import run as run_multimetric_report


class FoldBenchRnaRmsdRunnerTests(unittest.TestCase):
    def test_discover_candidates_reads_real_layout_and_ranking_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = (
                root
                / "predictions"
                / "pred_output_1abc_seed_42"
                / "1abc"
                / "seed_42"
                / "predictions"
            )
            pred.mkdir(parents=True)
            refs = root / "references"
            refs.mkdir()
            (refs / "1abc.cif").write_text("data_1abc\n", encoding="utf-8")
            for sample, score in ((0, 0.75), (1, 0.90)):
                (pred / f"1abc_sample_{sample}.cif").write_text(
                    "data_prediction\n", encoding="utf-8"
                )
                (pred / f"1abc_summary_confidence_sample_{sample}.json").write_text(
                    json.dumps({"ranking_score": score}), encoding="utf-8"
                )
            # This secondary file must not be counted as a primary prediction.
            (pred / "1abc_sample_0_wounresol.cif").write_text(
                "data_secondary\n", encoding="utf-8"
            )

            candidates, issues = discover_candidates(root / "predictions", refs)

            self.assertEqual(len(candidates), 2)
            self.assertEqual(issues, [])
            self.assertEqual(candidates[0].pdb_id, "1ABC")
            self.assertEqual(candidates[0].seed, 42)
            self.assertEqual(candidates[1].sample, 1)
            self.assertAlmostEqual(candidates[1].ranking_score, 0.90)
            self.assertTrue(candidates[0].reference_path.endswith("1abc.cif"))

            limited, _ = discover_candidates(
                root / "predictions", refs, max_candidates=1
            )
            self.assertEqual(len(limited), 1)

            manifest = root / "candidates.csv"
            write_manifest(candidates, manifest)
            reloaded = read_manifest(manifest)
            self.assertEqual(reloaded, candidates)

    def test_valid_output_requires_success_and_finite_rmsd(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(
                json.dumps({"status": "SUCCESS", "rmsd": 3.165}), encoding="utf-8"
            )
            payload, issue = load_valid_ost_output(path)
            self.assertIsNotNone(payload)
            self.assertEqual(issue, "")

            path.write_text(
                json.dumps({"status": "SUCCESS", "rmsd": None}), encoding="utf-8"
            )
            payload, issue = load_valid_ost_output(path)
            self.assertIsNone(payload)
            self.assertIn("finite", issue)


class FoldBenchRnaRmsdSummaryTests(unittest.TestCase):
    def test_audited_lddt_failure_can_use_separate_rigid_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = Candidate(
                pdb_id="1ABC",
                seed=42,
                sample=0,
                prediction_path="prediction.cif",
                reference_path="reference.cif",
                confidence_path="confidence.json",
                ranking_score=0.9,
            )
            manifest = root / "manifest" / "candidates.csv"
            write_manifest([candidate], manifest)
            error = root / "errors" / "1ABC" / "seed_42" / "sample_0.stderr.txt"
            error.parent.mkdir(parents=True)
            error.write_text(
                "Computing all-atom lDDT\n"
                "ValueError: need at least one array to concatenate\n",
                encoding="utf-8",
            )
            self.assertTrue(has_audited_lddt_failure(root, candidate))

            rescue = (
                root
                / "rigid_only_rescue"
                / "details"
                / "1ABC"
                / "seed_42"
                / "sample_0.json"
            )
            rescue.parent.mkdir(parents=True)
            rescue.write_text(
                json.dumps({"status": "SUCCESS", "rmsd": 12.5}),
                encoding="utf-8",
            )

            frame = load_candidate_results(root, manifest)
            self.assertEqual(frame.loc[0, "eval_status"], "SUCCESS")
            self.assertEqual(frame.loc[0, "evaluation_protocol"], "rigid_only_rescue")
            self.assertAlmostEqual(frame.loc[0, "rmsd"], 12.5)
            self.assertTrue(np.isnan(frame.loc[0, "lddt"]))

    def test_downstream_tm_failure_after_rmsd_is_rescuable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = Candidate(
                pdb_id="3P4A",
                seed=42,
                sample=0,
                prediction_path="prediction.cif",
                reference_path="reference.cif",
                confidence_path="confidence.json",
                ranking_score=0.8,
            )
            error = root / "errors" / "3P4A" / "seed_42" / "sample_0.stderr.txt"
            error.parent.mkdir(parents=True)
            error.write_text(
                "stdout='ERROR! No assignable chain\\n'; "
                "stderr='Computing RMSD\\nComputing patch TM-score with USalign exectuable\\n'",
                encoding="utf-8",
            )
            self.assertEqual(
                audited_rescue_reason(root, candidate),
                "DOWNSTREAM_TM_NO_ASSIGNABLE_CHAIN_AFTER_RMSD",
            )

    def test_rank1_never_falls_back_to_successful_lower_rank(self):
        frame = pd.DataFrame(
            [
                {
                    "pdb_id": "1ABC",
                    "seed": 42,
                    "sample": 0,
                    "ranking_score": 0.95,
                    "eval_status": "missing_output",
                    "rmsd": np.nan,
                },
                {
                    "pdb_id": "1ABC",
                    "seed": 66,
                    "sample": 0,
                    "ranking_score": 0.90,
                    "eval_status": "SUCCESS",
                    "rmsd": 1.25,
                },
            ]
        )

        rank1 = select_strict_rank1(frame)

        self.assertEqual(len(rank1), 1)
        self.assertEqual(rank1.iloc[0]["seed"], 42)
        self.assertTrue(np.isnan(rank1.iloc[0]["rmsd"]))
        self.assertAlmostEqual(rank1.iloc[0]["oracle_min_rmsd"], 1.25)

    def test_describe_rmsd_keeps_missing_values_in_denominator(self):
        frame = pd.DataFrame({"rmsd": [1.0, 3.0, np.nan, 11.0]})
        result = describe_rmsd(frame, label="test")
        self.assertEqual(result["n_total"], 4)
        self.assertEqual(result["n_with_rmsd"], 3)
        self.assertEqual(result["n_missing_rmsd"], 1)
        self.assertAlmostEqual(result["rmsd_median_angstrom"], 3.0)
        self.assertAlmostEqual(result["rmsd_le_2A_percent"], 100 / 3)

    def test_rna_input_size_uses_sequence_length_times_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1abc.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "sequences": [
                                {"rnaSequence": {"sequence": "AUCG", "count": 2}},
                                {"rnaSequence": {"sequence": "AAA", "count": 1}},
                                {"ligand": {"ligand": "CCD_MG", "count": 3}},
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            total, chains, issue = rna_input_size(path)
            self.assertEqual(total, 11)
            self.assertEqual(chains, 3)
            self.assertEqual(issue, "")

    def test_read_ost_result_accepts_missing_rigid_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "SUCCESS",
                        "rmsd": 3.165,
                        "lddt": 0.425,
                        "tm_score": 0.149,
                    }
                ),
                encoding="utf-8",
            )
            result = read_ost_result(path)
            self.assertEqual(result["eval_status"], "SUCCESS")
            self.assertAlmostEqual(result["rmsd"], 3.165)

    def test_end_to_end_summary_writes_primary_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "rmsd"
            manifest_dir = output / "manifest"
            manifest_dir.mkdir(parents=True)
            rows = []
            for sample, score, rmsd in ((0, 0.8, 4.0), (1, 0.9, 2.0)):
                rows.append(
                    {
                        "pdb_id": "1ABC",
                        "seed": 42,
                        "sample": sample,
                        "prediction_path": f"pred_{sample}.cif",
                        "reference_path": "1abc.cif",
                        "confidence_path": f"confidence_{sample}.json",
                        "ranking_score": score,
                        "discovery_issue": "",
                    }
                )
                detail = output / "details" / "1ABC" / "seed_42"
                detail.mkdir(parents=True, exist_ok=True)
                (detail / f"sample_{sample}.json").write_text(
                    json.dumps(
                        {
                            "status": "SUCCESS",
                            "rmsd": rmsd,
                            "lddt": 0.5,
                            "tm_score": 0.4,
                        }
                    ),
                    encoding="utf-8",
                )
            pd.DataFrame(rows).to_csv(manifest_dir / "candidates.csv", index=False)

            targets = root / "targets.xlsx"
            with pd.ExcelWriter(targets, engine="openpyxl") as writer:
                pd.DataFrame(
                    [
                        {
                            "PDB_id": "1ABC",
                            "release_date": "2022-01-01",
                            "time_group": "post_cutoff",
                            "chain_count": 1,
                        }
                    ]
                ).to_excel(writer, sheet_name="Targets", index=False)
            simple_json = root / "Simple_json"
            simple_json.mkdir()
            (simple_json / "1abc.json").write_text(
                json.dumps(
                    [
                        {
                            "sequences": [
                                {"rnaSequence": {"sequence": "AUCG", "count": 1}}
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )

            exit_code = run_summary(
                argparse.Namespace(
                    output_root=str(output),
                    manifest=None,
                    report_dir=None,
                    targets_xlsx=str(targets),
                    simple_json_root=str(simple_json),
                )
            )

            self.assertEqual(exit_code, 0)
            report = output / "reports"
            self.assertTrue((report / "rmsd_report.xlsx").is_file())
            self.assertTrue((report / "rmsd_overview.png").is_file())
            self.assertTrue((report / "pdb_rmsd_audit.tsv").is_file())
            self.assertTrue(
                (report / "exclude_strict_rank1_rmsd_pdb.txt").is_file()
            )
            rank1 = pd.read_csv(report / "rank1_targets.csv")
            self.assertEqual(rank1.loc[0, "sample"], 1)
            self.assertAlmostEqual(rank1.loc[0, "rmsd"], 2.0)

    def test_final_report_honors_frozen_exclusions_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rmsd"
            reports = root / "reports"
            reports.mkdir(parents=True)
            pdb_ids = ["1A01", "1A02", "1A03", "1A04", "1A05", "1A06"]
            lengths = [10, 30, 70, 150, 250, 30]
            length_groups = ["1-20", "21-50", "51-100", "101-200", ">200", "21-50"]
            chain_groups = ["1", "2", "3", "4", ">=5", "2"]
            rmsds = [1.0, 2.0, 4.0, 8.0, 16.0, np.nan]
            rank_rows = []
            candidate_rows = []
            for index, pdb_id in enumerate(pdb_ids):
                status = "FAILURE" if pdb_id == "1A06" else "SUCCESS"
                rank_rows.append(
                    {
                        "pdb_id": pdb_id,
                        "seed": 42,
                        "sample": 0,
                        "ranking_score": 0.9,
                        "rmsd": rmsds[index],
                        "eval_status": status,
                        "time_group": "pre_or_on_cutoff" if index < 3 else "post_cutoff",
                        "rna_total_length": lengths[index],
                        "length_group": length_groups[index],
                        "chain_count": 5 if chain_groups[index] == ">=5" else int(chain_groups[index]),
                        "chain_count_group": chain_groups[index],
                        "prediction_path": f"/pred/{pdb_id}.cif",
                        "reference_path": f"/ref/{pdb_id}.cif",
                    }
                )
                candidate_rows.append(
                    {
                        "pdb_id": pdb_id,
                        "seed": 42,
                        "sample": 0,
                        "rmsd": rmsds[index],
                        "eval_status": status,
                    }
                )
            candidate_rows.append(
                {
                    "pdb_id": "1A06",
                    "seed": 66,
                    "sample": 1,
                    "rmsd": 3.0,
                    "eval_status": "SUCCESS",
                }
            )
            pd.DataFrame(rank_rows).to_csv(reports / "rank1_targets.csv", index=False)
            pd.DataFrame(candidate_rows).to_csv(reports / "all_candidates.csv", index=False)
            pd.DataFrame(
                [{"pdb_id": "1A06", "exclusion_reason": "STRICT_RANK1_RMSD_FAILED_BUT_LOWER_CANDIDATE_AVAILABLE"}]
            ).to_csv(reports / "pdb_rmsd_audit.tsv", sep="\t", index=False)
            (reports / "exclude_strict_rank1_rmsd_pdb.txt").write_text(
                "1A06\n", encoding="utf-8"
            )
            error = root / "errors" / "1A06" / "seed_42" / "sample_0.stderr.txt"
            error.parent.mkdir(parents=True)
            error.write_text(
                "Computing GDT-TS score\nValueError: window size is too large\n",
                encoding="utf-8",
            )

            exit_code = run_final_report(
                argparse.Namespace(
                    rmsd_root=str(root),
                    exclusion_list=None,
                    output_dir=None,
                    expected_targets=6,
                    expected_excluded=1,
                    expected_valid=5,
                )
            )

            self.assertEqual(exit_code, 0)
            final = root / "final_report"
            self.assertTrue((final / "figures/Figure1_RMSD_ECDF.png").is_file())
            self.assertTrue((final / "figures/Figure2_RMSD_stratified_boxplots.pdf").is_file())
            self.assertTrue((final / "RNA_RMSD_final_report.xlsx").is_file())
            self.assertEqual(
                (final / "frozen_excluded_pdb_ids.txt").read_text(encoding="utf-8"),
                "1A06\n",
            )
            main = pd.read_csv(final / "tables/Table1_main_RMSD_results.tsv", sep="\t")
            self.assertEqual(int(main.loc[0, "n_total"]), 6)
            self.assertEqual(int(main.loc[0, "n_valid"]), 5)
            supplement = pd.read_csv(final / "tables/TableS1_excluded_PDB.tsv", sep="\t")
            self.assertEqual(supplement.loc[0, "failure_stage"], "RIGID_GDT")
            self.assertEqual(supplement.loc[0, "standard_result"], "FAILED")
            self.assertTrue(bool(supplement.loc[0, "lower_rank_valid_candidate_available"]))

    def test_multimetric_report_uses_metric_specific_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rmsd"
            reports = root / "reports"
            reports.mkdir(parents=True)
            rows = []
            lengths = [10, 30, 70, 150, 250, 35]
            length_groups = ["1-20", "21-50", "51-100", "101-200", ">200", "21-50"]
            chain_groups = ["1", "2", "3", "4", ">=5", "2"]
            for index in range(6):
                rows.append(
                    {
                        "pdb_id": f"2A0{index + 1}",
                        "seed": 42,
                        "sample": 0,
                        "ranking_score": 0.9 - index * 0.01,
                        "time_group": "pre_or_on_cutoff" if index < 3 else "post_cutoff",
                        "rna_total_length": lengths[index],
                        "length_group": length_groups[index],
                        "chain_count_group": chain_groups[index],
                        "lddt": np.nan if index == 5 else 0.82 - index * 0.09,
                        "tm_score": np.nan if index == 5 else 0.77 - index * 0.08,
                        "oligo_gdtts": 0.72 - index * 0.07,
                        "rmsd": float(1 + index * 3),
                        "prediction_path": f"/pred/2A0{index + 1}.cif",
                        "reference_path": f"/ref/2A0{index + 1}.cif",
                        "evaluation_protocol": "rigid_only_rescue" if index == 5 else "foldbench_full",
                        "eval_status": "SUCCESS",
                        "eval_issue": "",
                    }
                )
            pd.DataFrame(rows).to_csv(reports / "rank1_targets.csv", index=False)

            exit_code = run_multimetric_report(
                argparse.Namespace(
                    rmsd_root=str(root),
                    output_dir=None,
                    expected_targets=6,
                    bootstrap_replicates=100,
                )
            )

            self.assertEqual(exit_code, 0)
            output = root.parent / "foldbench_style_multimetric_report"
            self.assertTrue((output / "figures/Figure1_primary_LDDT.png").is_file())
            self.assertTrue((output / "figures/Figure2_metrics_by_length_and_chain_count.pdf").is_file())
            self.assertTrue((output / "figures/Figure3_local_vs_global_accuracy.svg").is_file())
            self.assertTrue((output / "FoldBench_style_RNA_multimetric_report.xlsx").is_file())
            summary = pd.read_csv(output / "tables/Table2_all_metrics.tsv", sep="\t")
            lddt = summary[(summary["group"] == "All targets") & (summary["metric"] == "lddt")].iloc[0]
            rmsd = summary[(summary["group"] == "All targets") & (summary["metric"] == "rmsd")].iloc[0]
            self.assertEqual(int(lddt["n_valid"]), 5)
            self.assertEqual(int(rmsd["n_valid"]), 6)


if __name__ == "__main__":
    unittest.main()
