# Protenix RNA FoldBench-style RMSD evaluation

This pipeline evaluates every existing primary Protenix sample with the same
OpenStructure command options used by FoldBench 2.8, then selects one strict
rank-1 candidate per target using maximum `ranking_score`.

## Metric contract

- OpenStructure version remains pinned at 2.8.0.
- Each primary prediction is evaluated against `~/pdb_data/<PDB>.cif`.
- The command uses `--fault-tolerant --min-pep-length 4 --min-nuc-length 4
  --lddt --rigid-scores --tm-score --dockq`.
- The reported rigid RMSD is the OpenStructure nucleic-acid C3' RMSD, not an
  all-heavy-atom RMSD.
- A result is reusable only when the JSON parses, has `status=SUCCESS`, and has
  a finite non-negative `rmsd`.
- Rank-1 is chosen strictly by maximum Protenix `ranking_score`.  Evaluation
  failure never causes fallback to a lower-ranked candidate.
- OpenStructure 2.8 does not serialize `rigid_chain_mapping`; its absence does
  not invalidate a successful RMSD result.

## Install the report-only dependencies

The runner itself uses only the Python standard library.  The report step also
needs openpyxl and matplotlib:

```bash
conda activate foldbench
python -m pip install -r ~/Code/predict_protenix/requirements_foldbench_rmsd_report.txt
```

## Launch

```bash
chmod +x ~/Code/predict_protenix/{start_foldbench_rna_rmsd,monitor_foldbench_rna_rmsd}.sh
```

First run 50 candidates with four workers:

```bash
bash ~/Code/predict_protenix/start_foldbench_rna_rmsd.sh smoke
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_rna_rmsd.sh
```

After smoke has exit code 0 and no failed candidates, start the full resumable
run.  The current shared-node load is high, so the default is deliberately 16
workers rather than all 192 logical CPUs:

```bash
bash ~/Code/predict_protenix/start_foldbench_rna_rmsd.sh run
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_rna_rmsd.sh
```

Valid smoke outputs are reused by the full run.  Repeating `run` after an
interruption or candidate failure is safe: validated JSON files are skipped and
only missing/invalid candidates are invoked again.  The first full run writes a
complete `manifest/candidates.csv`; later retries reuse it and therefore do not
rescan the prediction tree.  Use `--refresh-manifest` only if prediction files
were deliberately added or removed.  An advisory lock prevents two full
runners from writing to the same output root concurrently.

To override concurrency for a later retry:

```bash
RMSD_WORKERS=32 bash ~/Code/predict_protenix/start_foldbench_rna_rmsd.sh run
```

Do this only after the previous runner has exited.  With the observed load
average above 450, keep the initial full run at 16 workers.

## Summarize

If the standard command reports either the audited OpenStructure 2.8 all-atom
lDDT empty-array failure or a downstream USalign TM-score failure after RMSD
was reached, run the separate rigid-only rescue after the full runner completes.
It never overwrites standard outputs:

```bash
bash ~/Code/predict_protenix/start_foldbench_rna_rmsd.sh rescue
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_rna_rmsd.sh
```

The report selects standard successful JSON first and uses rescue RMSD only for
those audited failure classes.  Rescued rows have
`evaluation_protocol=rigid_only_rescue`; their lDDT and TM-score remain missing.

Then generate the report:

```bash
bash ~/Code/predict_protenix/start_foldbench_rna_rmsd.sh summarize
watch -n 30 bash ~/Code/predict_protenix/monitor_foldbench_rna_rmsd.sh
```

Stable outputs are under:

```text
~/Json_data/Foldbench_evaluation/rmsd/
  manifest/candidates.csv
  details/<PDB>/seed_<SEED>/sample_<N>.json
  errors/<PDB>/seed_<SEED>/sample_<N>.stderr.txt
  progress.json
  run_summary.json
  reports/all_candidates.csv
  reports/rank1_targets.csv
  reports/failed_candidates.tsv
  reports/rmsd_report.xlsx
  reports/rmsd_overview.png
  reports/rmsd_ecdf_full_log.png
  reports/run_summary.txt
```

`rmsd_report.xlsx` contains overall rank-1 statistics, all-candidate and oracle
comparisons, release-time groups, total-RNA-length groups, GT RNA-chain-count
groups, all rank-1 rows, all candidate rows, and failures.  Counts missing a
metric stay visible in every denominator.
