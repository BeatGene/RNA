"""Promote the existing rank-1 val/test symlinks into Data_PT_V2.

The complete val/test candidate pools are moved to a sibling archive. Selected
PT files are hard-linked into new val/test directories at their original paths,
so existing Data_PT_V2_rank1 symlinks continue to resolve. Train is untouched.
The symlinks themselves are the sole source of selection; scores are not read
or recalculated.

Run without --execute to audit only. No candidates are deleted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import shutil
import sys


SPLITS = ("val", "test")
SEED = re.compile(r"seed_(\d+)")
SAMPLE = re.compile(r"sample_(\d+)\.pt")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def identity(path: Path, root: Path, split: str) -> tuple[str, int, int]:
    relative = path.relative_to(root / split)
    require(len(relative.parts) == 3, f"unexpected PT layout: {path}")
    pdb, seed_dir, sample_file = relative.parts
    seed, sample = SEED.fullmatch(seed_dir), SAMPLE.fullmatch(sample_file)
    require(bool(pdb and seed and sample), f"unexpected PT name: {path}")
    return pdb.upper(), int(seed[1]), int(sample[1])


def audit(pt_root: Path, rank_root: Path, archive_root: Path,
          stage_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    require(pt_root.is_dir() and rank_root.is_dir(), "PT or rank-1 root is missing")
    require((pt_root / "train").is_dir(), "Data_PT_V2/train is missing")
    require(not archive_root.exists() and not archive_root.is_symlink(),
            f"archive path already exists: {archive_root}")
    require(not stage_root.exists() and not stage_root.is_symlink(),
            f"stage path already exists: {stage_root}")
    require(pt_root.parent == archive_root.parent == stage_root.parent,
            "PT, archive and stage must be siblings on the same filesystem")
    require(os.stat(pt_root).st_dev == os.stat(pt_root.parent).st_dev,
            "Data_PT_V2 is a mount point; directory renames would cross filesystems")

    picked: dict[str, list[tuple[Path, Path]]] = {}
    for split in SPLITS:
        original_dir, rank_dir = pt_root / split, rank_root / split
        require(original_dir.is_dir() and rank_dir.is_dir(),
                f"missing {split} directory")
        candidates = sorted(original_dir.rglob("*.pt"))
        rank_files = sorted(rank_dir.rglob("*.pt"))
        require(bool(candidates) and bool(rank_files), f"empty {split} PT data")
        counts = Counter(identity(path, pt_root, split)[0] for path in candidates)
        require(all(path.is_file() and not path.is_symlink() for path in candidates),
                f"{split} candidates include links or missing files")
        require(len(rank_files) == len(counts),
                f"{split}: rank-1 count differs from PDB count")

        picked[split] = []
        seen_pdb = set()
        for link in rank_files:
            pdb, seed, sample = identity(link, rank_root, split)
            require(pdb not in seen_pdb, f"multiple rank-1 PTs for {split}/{pdb}")
            seen_pdb.add(pdb)
            require(link.is_symlink(), f"rank-1 PT is not a symlink: {link}")
            source = pt_root / split / link.relative_to(rank_dir)
            require(source.is_file() and not source.is_symlink(),
                    f"selected source PT is missing or linked: {source}")
            require(link.resolve(strict=True) == source.resolve(strict=True),
                    f"rank-1 link does not target Data_PT_V2: {link}")
            picked[split].append((source, link.relative_to(rank_dir)))
        require(seen_pdb == set(counts),
                f"{split}: rank-1 links do not cover every candidate PDB")

        incomplete = sum(count != 200 for count in counts.values())
        print(f"AUDIT {split}: {len(candidates)} source PTs, "
              f"{len(picked[split])} PDBs, {incomplete} without exactly 200 PTs")
    print(f"ARCHIVE {archive_root}")
    print(f"STAGE {stage_root}")
    return picked


def execute(pt_root: Path, rank_root: Path, archive_root: Path,
            stage_root: Path, picked: dict[str, list[tuple[Path, Path]]]) -> None:
    stage_root.mkdir()
    try:
        for split in SPLITS:
            for source, relative in picked[split]:
                destination = stage_root / split / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, destination)
                require(os.path.samefile(source, destination),
                        f"staged hard link differs from source: {destination}")
            require(len(list((stage_root / split).rglob("*.pt"))) == len(picked[split]),
                    f"staging count mismatch: {split}")

        archive_root.mkdir()
        for name in ("selection.tsv", "coverage.tsv", "summary.json"):
            if (rank_root / name).is_file():
                shutil.copy2(rank_root / name, archive_root / f"rank1_{name}")

        archived, promoted = set(), set()
        try:
            for split in SPLITS:
                os.replace(pt_root / split, archive_root / split)
                archived.add(split)
                os.replace(stage_root / split, pt_root / split)
                promoted.add(split)
            for split in SPLITS:
                require(len(list((pt_root / split).rglob("*.pt"))) == len(picked[split]),
                        f"promoted count mismatch: {split}")
                for _, relative in picked[split]:
                    current = pt_root / split / relative
                    archived_source = archive_root / split / relative
                    rank_link = rank_root / split / relative
                    require(current.is_file() and
                            rank_link.resolve(strict=True) == current.resolve(strict=True),
                            f"rank-1 link broke after promotion: {rank_link}")
                    require(os.path.samefile(current, archived_source),
                            f"promoted PT differs from archive: {current}")
        except BaseException:
            for split in reversed(SPLITS):
                if split in promoted:
                    os.replace(pt_root / split, stage_root / split)
                if split in archived:
                    os.replace(archive_root / split, pt_root / split)
            raise
        stage_root.rmdir()
        print("SUCCESS: Data_PT_V2 val/test now contain one rank-1 PT per PDB")
        print(f"ALL_CANDIDATES_ARCHIVED: {archive_root}")
    except BaseException:
        print(f"FAILED: inspect stage={stage_root} archive={archive_root}", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path.home()
    parser.add_argument("--pt-root", type=Path, default=home / "Data_PT_V2")
    parser.add_argument("--rank1-root", type=Path, default=home / "Data_PT_V2_rank1")
    parser.add_argument("--archive-root", type=Path,
                        default=home / "Data_PT_V2_all_candidates_20261013")
    parser.add_argument("--stage-root", type=Path,
                        default=home / "Data_PT_V2_rank1_stage_20261013")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    roots = [path.expanduser().absolute() for path in
             (args.pt_root, args.rank1_root, args.archive_root, args.stage_root)]
    require(len(set(roots)) == 4, "all four root paths must differ")
    picked = audit(*roots)
    if args.execute:
        execute(*roots, picked)
    else:
        print("DRY_RUN_OK: no files changed; rerun with --execute to promote")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
