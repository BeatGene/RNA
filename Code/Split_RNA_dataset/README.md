# Pure-RNA PDB chronological split

This pipeline processes the 2246 `*.cif` files in `~/pdb_data`, excludes the
five PDB IDs listed in the configured XLSX, and assigns all remaining 2241 PDB
entries as follows:

- train: initial PDB release date through 2023-12-31;
- val: 2024-01-01 through 2024-12-31;
- test: 2025-01-01 onward.

Test RNA entities are compared against train+val and then clustered internally.
A hit requires at least 80% sequence identity, 80% query coverage, and 80%
target coverage.  A homologous entity and every chain instantiated from that
entity are masked from evaluation; other entities/chains in the same PDB stay
evaluable.  A test PDB directory is omitted only when none of its RNA chains
remain evaluable.

Because nucleotide seed searches may miss extremely short polymers, pairs in
which either RNA entity is shorter than 15 nt are additionally checked with a
deterministic global edit-distance alignment.  The hit reports identify the
alignment source in `ALIGNMENT_SOURCE`.

The script never copies or deletes CIF files.  Execute mode only creates empty
directories such as `~/Data/train/157d`.

The chain-level evaluation contract is recorded in two complementary files:

- `test_chain_evaluation.tsv` is the human-readable audit table, one row per
  RNA chain;
- `test_evaluation_mask.json` is the machine-readable mask for downstream RMSD
  and other metric code.  Only chains with `evaluate: true` / status `EVALUATE`
  should contribute to metrics.

For example, if chain A belongs to a reference-homologous entity while B and C
are novel, the test PDB is retained, A is marked `MASK_REFERENCE_HOMOLOG`, and
B/C are marked `EVALUATE`.

## Server prerequisites

```bash
python3 -m pip install --user gemmi openpyxl
mmseqs version
```

If `mmseqs` is unavailable and Conda is installed:

```bash
conda install -c conda-forge -c bioconda mmseqs2
```

## 1. Dry run

Run this from the laboratory server.  No directories below `~/Data` are
created in this mode.

```bash
python3 ~/Code/Split_RNA_dataset/split_rna_dataset.py
```

The command prints the timestamped report directory created under
`~/Code/pipeline_reports`.  Review at least `summary.json`,
`test_chain_evaluation.tsv`, `test_pdb_evaluation.tsv`,
`test_evaluation_mask.json`, and `final_manifest.tsv`.

## 2. Create the empty directories

```bash
python3 ~/Code/Split_RNA_dataset/split_rna_dataset.py --execute
```

The command refuses to proceed if it detects an existing four-character PDB
directory under the wrong split.  Existing correctly assigned directories and
their contents are left untouched.

Every run stores its configuration, input checksums, parsed release dates,
entity sequences, raw MMseqs2 working files, filtered homology hits, final
manifest, directory actions, summary, and log in its report directory.
