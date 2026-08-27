# Data_V1 single-chain RNA split

This pipeline implements the agreed split:

- train: PDB release date through 2021-09-30, with valid strict rank-1 C3' RMSD <= 15 A;
- val: 2021-10-01 through 2023-12-31, without an RMSD cutoff;
- test: 2024-01-01 onward, without an RMSD cutoff;
- all splits contain only PDB entries with exactly one RNA chain;
- test is filtered against final train+val at 80% sequence identity, 80% query coverage, and 80% target coverage, then internally de-redundified;
- directory names are lower-case PDB IDs.

The default server inputs are:

```text
~/pdb_data
~/Code/Download_PDB_RAW/Second_PDB_ID_xlsx_and_InOut/alignment_pdb_ids_not_in_experimental_pure_rna.xlsx
~/Json_data/Foldbench_evaluation/rmsd/reports/rank1_targets.csv
~/Json_data/Foldbench_evaluation/rmsd/reports/exclude_strict_rank1_rmsd_pdb.txt
```

Install/check prerequisites in the Python environment used for the run:

```bash
python3 -m pip install --user gemmi openpyxl numpy matplotlib
mmseqs version
```

## Background dry-run

```bash
bash ~/Code/Split_RNA_dataset/start_split_rna_dataset_v1.sh dry-run
```

The launcher prints its PID and console-log path. It also updates:

```text
~/Code/pipeline_reports/current_data_split_v1_run.txt
```

Dry-run creates a complete report but does not create anything below
`~/Data_V1`.

## Background execute run

After reviewing the dry-run `summary.json` and `final_manifest.tsv`:

```bash
bash ~/Code/Split_RNA_dataset/start_split_rna_dataset_v1.sh execute
```

Execute mode creates empty directories only:

```text
~/Data_V1/train/<lowercase_pdb_id>/
~/Data_V1/val/<lowercase_pdb_id>/
~/Data_V1/test/<lowercase_pdb_id>/
```

It refuses to run if a four-character PDB directory already exists under the
wrong split. Existing correctly assigned directories and their contents are
left untouched.

All defaults can be overridden after the mode argument. For example:

```bash
bash ~/Code/Split_RNA_dataset/start_split_rna_dataset_v1.sh dry-run \
  --rank1-csv /another/path/rank1_targets.csv \
  --data-dir /another/path/Data_V1
```
