# Rank-1 PT promotion (2026-10-13)

`Data_PT_V2/train` remains unchanged. `Data_PT_V2/val` contains 103 PT files
and `Data_PT_V2/test` contains 67, one per PDB. The choices come directly from
the existing `Data_PT_V2_rank1` symlinks; no scores were recomputed. The prior
complete candidate directories were moved to
`~/Data_PT_V2_all_candidates_20261013/{val,test}` (19,722 and 12,310 PT files).
The archive also contains `rank1_selection.tsv`, `rank1_coverage.tsv`, and
`rank1_summary.json`.

The three `run_rank1_epoch49_*.sh` scripts now default to `~/Data_PT_V2` for
PT data and to the archive for the selection records. They use new default
output directories so historical results tied to `Data_PT_V2_rank1` stay
separate. `DATA_ROOT` and `RANK1_ARCHIVE_ROOT` can override these defaults.

Once the updated scripts are on the server, the old `Data_PT_V2_rank1` folder
can be moved to the cleanup hold. Its PT entries are symlinks, and deleting
them does not delete `Data_PT_V2` or the archived candidates. Do not rerun
`select_rank1_pt.py` against the promoted `Data_PT_V2` expecting the original
50-by-4 selection: each val/test PDB now has only its selected PT there.
