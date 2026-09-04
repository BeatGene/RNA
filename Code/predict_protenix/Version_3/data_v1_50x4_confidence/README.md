# Data_V1 Protenix 50x4 + full confidence

This run covers the 988 `FINAL_STATUS=KEPT` single-chain RNA targets in
`Data_V1`: train 774, val 117, and test 97.

Fixed inference settings:

- `protenix_base_default_v1.0.0`
- seeds 300 through 349, four samples per seed (200 structures per target)
- 200 diffusion steps, 10 Pairformer cycles, BF16
- `need_atom_confidence=true`
- each `*_full_data_sample_*.json` must contain `atom_plddt`,
  `token_pair_pae`, `token_pair_pde`, `contact_probs`, and
  `atom_to_token_idx`

Output layout:

```text
~/Data_V1/<train|val|test>/<pdb_id>/seed_<seed>/predictions/
```

The complete run expects 197,600 CIFs, summary-confidence JSON files, and
full-data JSON files. Full pairwise confidence JSON can consume substantial
disk space, so the launcher and worker stop before free space falls below 200
GiB by default.

Deploy the two modified shared files and this directory under `~/Code`, then
run:

```bash
bash ~/Code/predict_protenix/Version_3/data_v1_50x4_confidence/start_data_v1_50x4_confidence_8gpu.sh
```

Monitor the current run:

```bash
bash ~/Code/predict_protenix/Version_3/data_v1_50x4_confidence/monitor_data_v1_50x4_confidence.sh
```

The workflow is resumable: rerunning the start script audits every seed and
only schedules missing or invalid outputs. A seed is complete only when all
four CIFs, all four summary-confidence JSON files, and all four valid
full-data JSON files are present.

Before launching the background run, the start script performs a foreground
`1 target x 1 seed x 4 samples` smoke test and checks the five full-data keys.
Set `SMOKE_TEST_FIRST=0` only when that exact smoke test has already passed.

Resource thresholds and scheduling can be overridden as environment
variables, for example:

```bash
GPU_LIST=0,1,2,3 SPLIT_ORDER="val test" MIN_FREE_DISK_GIB=500 \
  bash ~/Code/predict_protenix/Version_3/data_v1_50x4_confidence/start_data_v1_50x4_confidence_8gpu.sh
```
