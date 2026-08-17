# RNA-FM pipeline utilities

This directory contains project-specific scripts for auditing inputs and later
generating RNA-FM residue embeddings.  It is separate from the cloned upstream
`ml4bio/RNA-FM` repository.

## Input audit on the laboratory server

The default arguments match the server layout under
`/storage9920/home/tinghao.xia`:

```bash
conda activate rna_fm
python /storage9920/home/tinghao.xia/Code/RNA_FM_pipeline/audit_rnafm_inputs.py
```

The script reads only:

- `Json_data/Simple_json/XXXX.json` (not `*-final-updated.json`)
- `Code/pipeline_reports/PDB_RAW/rna_chain_sequences.csv`
- the directory names under `Data/train`, `Data/val`, and `Data/test`
- the latest executed split report's `final_manifest.tsv` (auto-discovered)

The split manifest is used to distinguish an accidental missing directory from
a PDB intentionally removed as `DROP_NO_EVALUABLE_CHAINS`.  To select it
explicitly:

```bash
python /storage9920/home/tinghao.xia/Code/RNA_FM_pipeline/audit_rnafm_inputs.py \
  --split-manifest /storage9920/home/tinghao.xia/Code/pipeline_reports/DATA_SPLIT_2241_CHAINMASK_20260807T114307Z_EXECUTE/final_manifest.tsv
```

Non-ACGU symbols and chains longer than 1022 residues are warnings by default.
They remain visible in `chain_audit.tsv`; the embedding stage must map unsupported
symbols to the model's `<unk>` token and process long chains with overlapping
windows.  Use `--non-acgu-severity ERROR` or `--overlength-severity ERROR` for a
strict audit.

It writes a new timestamped directory under `Code/pipeline_reports`, for example:

```text
RNA_FM_AUDIT_20260811T130000Z/
  audit.log
  summary.json
  pdb_audit.tsv
  chain_audit.tsv
  issues.tsv
```

A successful audit exits with code 0 and prints `FINAL STATUS: PASS`.
`NEEDS_REVIEW` exits with code 1.  A fatal configuration/read error exits with
code 2.

## Runtime probe

After the input audit passes, validate the exact installed RNA-FM tokenizer,
maximum context length, GPU forward pass, and embedding dimension:

```bash
conda activate rna_fm
python /storage9920/home/tinghao.xia/Code/RNA_FM_pipeline/probe_rnafm_runtime.py
```

The probe tests `ACGUNXIT`, runs a normal forward pass, then runs one forward
pass at the model's maximum inferred residue length.  It writes
`probe.log` and `runtime_probe.json` under a timestamped
`Code/pipeline_reports/RNA_FM_RUNTIME_PROBE_*` directory.  It does not write
embeddings or modify dataset inputs.

## Residue embedding generation

Run a smoke test first.  These IDs cover `N`, `X`, `I`, `T`, multiple chains,
and a sequence longer than the 1022-residue model limit:

```bash
python /storage9920/home/tinghao.xia/Code/RNA_FM_pipeline/generate_rnafm_embeddings.py \
  --audit-report /storage9920/home/tinghao.xia/Code/pipeline_reports/RNA_FM_AUDIT_20260811T141902Z \
  --device cuda:0 \
  --pdb-id 1G2J 21ET 1SAQ 2LVY 7ZFW
```

After inspecting the smoke outputs, run all audited PDBs by omitting
`--pdb-id`.  Existing valid files are verified and skipped, so the smoke files
are reused and interrupted runs are resumable.

Outputs are centralized under:

```text
Data_FM/RNA_FM_embeddings/<PDB_ID>/rnafm_t12_residue_embeddings.pt
```

Each PDB file stores all chains in JSON-expanded order.  Important tensors are
`residue_embedding [R,640]`, `chain_offsets [C+1]`,
`residue_chain_index [R]`, `residue_index_in_chain [R]`,
`residue_token_id [R]`, `residue_non_acgu [R]`, and
`residue_is_unk [R]`.  Long chains use 1022-residue overlapping windows and a
uniform mean in overlap regions.  Embeddings are stored as float32 by default.

Validate the smoke outputs before a full run:

```bash
python /storage9920/home/tinghao.xia/Code/RNA_FM_pipeline/validate_rnafm_embeddings.py \
  --audit-report /storage9920/home/tinghao.xia/Code/pipeline_reports/RNA_FM_AUDIT_20260811T141902Z \
  --pdb-id 1G2J 21ET 1SAQ 2LVY 7ZFW
```

Omit `--pdb-id` after full generation to validate every audited PDB.  The
validator checks metadata, tensor shapes, finite values, residue/token/mask
alignment, identical-sequence chain equality, and long-window coverage.

## Local tests

```bash
python -m unittest discover -s Code/RNA_FM_pipeline/tests -v
```
