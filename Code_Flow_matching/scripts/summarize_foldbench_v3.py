"""Check FoldBench RNA monomer score coverage and compare paired lDDT values."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foldbench-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.foldbench_dir
    targets = read_csv(root / "targets" / "monomer_rna.csv")
    expected = {row["pdb_id"] for row in targets}
    if len(expected) != len(targets):
        raise ValueError("Duplicate FoldBench targets")
    by_algorithm = {}
    for algorithm in ("ProtenixV3Input", "ProtenixV3Refined"):
        rows = read_csv(root / "evaluation" / algorithm / "raw" / "monomer_rna_ost.csv")
        if len(rows) != len(expected):
            raise ValueError(f"{algorithm}: expected {len(expected)} scores, found {len(rows)}")
        indexed = {}
        for row in rows:
            target = row["pdb_id"]
            if target in indexed or target not in expected:
                raise ValueError(f"{algorithm}: duplicate or unexpected target {target}")
            value = float(row["lddt"])
            if not math.isfinite(value):
                raise ValueError(f"{algorithm}: invalid lDDT for {target}")
            indexed[target] = (value, row["seed"], row["sample"])
        by_algorithm[algorithm] = indexed
    rows = []
    for target in sorted(expected):
        before, before_seed, before_sample = by_algorithm["ProtenixV3Input"][target]
        after, after_seed, after_sample = by_algorithm["ProtenixV3Refined"][target]
        if (before_seed, before_sample) != (after_seed, after_sample):
            raise ValueError(f"Different input/refined candidate for {target}")
        rows.append({
            "foldbench_target_id": target,
            "seed": before_seed,
            "sample": before_sample,
            "input_lddt": before,
            "refined_lddt": after,
            "lddt_change": after - before,
        })
    with (root / "paired_lddt.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    input_mean = statistics.fmean(row["input_lddt"] for row in rows)
    refined_mean = statistics.fmean(row["refined_lddt"] for row in rows)
    lines = [
        "# FoldBench RNA monomer intersection: paired lDDT", "",
        f"Matched targets: {len(rows)} / 15 official monomer RNA targets", "",
        f"Mean lDDT: {input_mean:.4f} -> {refined_mean:.4f} (change {refined_mean - input_mean:+.4f})", "",
        "Each target uses the same candidate selected by original Protenix ranking_score; no native-based candidate selection.",
        "This is a subset and a post-processing comparison, not the complete official leaderboard score.", "",
        "| target | seed | sample | input lDDT | refined lDDT | change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['foldbench_target_id']} | {row['seed']} | {row['sample']} | "
            f"{row['input_lddt']:.4f} | {row['refined_lddt']:.4f} | {row['lddt_change']:+.4f} |"
        )
    (root / "paired_lddt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"FOLDBENCH_SCORE_COMPLETE targets={len(rows)} input_lddt={input_mean:.4f} refined_lddt={refined_mean:.4f}")


if __name__ == "__main__":
    main()
