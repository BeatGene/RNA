# RNA 数据集 V2：服务器执行顺序

此文档的路径均为实验室服务器路径。先把本次本地修改同步到服务器。旧的 `Data_V1`、`Data_PT_V1` 和旧报告只读保留；新输出使用 `Data_V2`、`Data_PT_V2`、`Data_PT_V2_RMSD_GT30` 和 `~/Code/pipeline_reports` 下的新类别目录。每一步检查完成后再执行下一步。

> 2026-09-25 更新：V2 EXECUTE 已完成。用户决定保留 V2 划分，但对 14 个因超长链放弃 Protenix prep 的 PDB 跳过后续任务。其中 4 个多 RNA 链本来就未入选；另外 9 个 train、1 个 val 已入选且正好是 prep 审计中缺失的 10 个。**暂勿照本文件旧的第 3、4 步对全部 1015 个 PDB 运行 prep 或 50×4 预测。**先使用 `prep_abandoned_long_chain_v2.tsv` 和 `audit_v2_prediction_reuse.py` 核对名单、审计旧版 964 个可复用预测文件；该脚本默认只读，`--execute-links` 应在审计报告通过后另行执行。后续预测清单仅含新增且 prep 可用的 41 个 PDB。第 5 步 PT 生成也须显式跳过那 10 个，避免把已知 prep 放弃误报成 PT 失败。

当前先将 `audit_v2_prediction_reuse.py`、`prep_abandoned_long_chain_v2.tsv` 和 `run_v2_prediction_reuse_audit.sh` 同步到服务器同目录，然后运行：

```bash
bash -n ~/Code/Split_RNA_dataset/Version_2/run_v2_prediction_reuse_audit.sh
bash ~/Code/Split_RNA_dataset/Version_2/run_v2_prediction_reuse_audit.sh
```

输出在 `~/Code/pipeline_reports/DATA_V2_PRED_REUSE_AUDIT_<时间>_DRYRUN/`。检查 `summary.json`：预期 `SKIP_PREP_ABANDONED_LONG_CHAIN=10`、`NEW_PREDICTION_REQUIRED=41`，其余 964 个应为 `REUSE_READY_TO_LINK` 或 `REUSE_ALREADY_LINKED`。这是文件名、非空文件和既有预测审计日志的核对；本阶段尚不创建链接，也不启动 GPU。若有 `REVIEW_BEFORE_PREDICTION`，先查看 `pdb_prediction_plan.tsv` 的具体原因。

2026-09-24T164258Z 的 dry run 已核对通过：964 个 `REUSE_READY_TO_LINK`、41 个 `NEW_PREDICTION_REQUIRED`、10 个 `SKIP_PREP_ABANDONED_LONG_CHAIN`，无 blocker。同步新版 `audit_v2_prediction_reuse.py` 和 `run_v2_prediction_reuse_link.sh` 后，运行 `bash -n` 和 `bash ~/Code/Split_RNA_dataset/Version_2/run_v2_prediction_reuse_link.sh`。该脚本仅移除 964 个已经审计为空的 V2 PDB 目录，并以指向 V1 PDB 目录的符号链接代替；每个链接均检查能否读取 200 个原始预测 CIF。遇到不一致会停止，不递归删除，也不修改 V1。链接报告应为 `REUSE_LINKED=964`、`NEW_PREDICTION_REQUIRED=41`、`SKIP_PREP_ABANDONED_LONG_CHAIN=10`。新版 `predict_all_pdb_ids.txt` 是后续仅调度 41 个新 PDB 的输入清单。符号链接不提供文件系统只读保护，因此后续预测程序必须仅接收 41 个新 PDB 的 manifest。

## 1. 记录旧版逐 PDB 状态

```bash
bash ~/Code/Split_RNA_dataset/Version_2/run_old_v1_lifecycle_audit.sh
```

查看 `summary.json` 和 `pdb_lifecycle.tsv`。旧版 61 个 PT 策略排除应与真正的 prep/pred/PT 失败分开。旧版被过滤的 >30 Å 构象只在日志里有记录，并无对应 PT。

## 2. 新版数据划分，先 dry run

```bash
python ~/Code/Split_RNA_dataset/Version_2/split_rna_dataset_v1.py --data-dir ~/Data_V2 --report-root ~/Code/pipeline_reports --rank1-annotation-only --threads 16 --mmseqs ~/miniconda3/envs/rna_pdb/bin/mmseqs
```

本步只生成 `DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_*_DRYRUN` 报告。检查 `summary.json`、`selection_audit.tsv`、`final_manifest.tsv` 和测试集同源命中表：2246 个源 CIF、5 个源排除、单链限制、原时间窗口、测试集对 train/val 去同源和测试集内部去冗余。缺失或无效 rank-1、冻结清单和训练集 rank-1 RMSD >15 Å 均只作注释。

确认 dry run 后，以相同参数增加 `--execute` 生成独立的 `~/Data_V2/{train,val,test}` 空目录和 `*_EXECUTE` 报告。后续在 shell 中把实际报告目录赋给变量，例如：

```bash
new_split=~/Code/pipeline_reports/DATA_SPLIT_V2_SINGLECHAIN_RANK1_ANNOTATION_<实际时间戳>_EXECUTE
```

## 3. 新入选 PDB 的 Protenix 输入准备

用新划分生成预测任务 CSV，并检查输入是否存在：

```bash
python ~/Code/Split_RNA_dataset/Version_2/prepare_new_split_protenix_inputs.py --split-report "$new_split" --output-dir ~/Code/pipeline_reports/DATA_V2_PREP/input_plan
tasks=~/Code/pipeline_reports/DATA_V2_PREP/input_plan/protenix_tasks.csv
protenix_py=~/miniconda3/envs/protenix-1.0.5/bin/python
$protenix_py ~/Code/predict_protenix/stage2_decoy_pipeline.py audit --manifest "$tasks" --report-dir ~/Code/pipeline_reports/DATA_V2_PREP/audit_before
```

若审计显示原始 JSON 或 prep 缺失，先在 Protenix 环境中对同一任务 CSV 运行 `make-json`，再运行 `prep`；两者只补缺失或损坏的输入。示例：

```bash
source ~/Code/predict_protenix/protenix_env.sh
$protenix_py ~/Code/predict_protenix/stage2_decoy_pipeline.py make-json --manifest "$tasks" --report-dir ~/Code/pipeline_reports/DATA_V2_PREP/make_json
$protenix_py ~/Code/predict_protenix/stage2_decoy_pipeline.py prep --manifest "$tasks" --report-dir ~/Code/pipeline_reports/DATA_V2_PREP/prep --workers 4 --nhmmer-cpus 8
$protenix_py ~/Code/predict_protenix/stage2_decoy_pipeline.py audit --manifest "$tasks" --report-dir ~/Code/pipeline_reports/DATA_V2_PREP/audit_after
```

完成后确认新入选 PDB 的 updated JSON 和 prep 均可用，再开始预测。

## 4. 在新目录生成 50×4 预测

新预测启动脚本已支持 `SPLIT_MANIFEST`、`DATA_ROOT`、`BASE_REPORT` 和 `ALLOW_VARIABLE_SPLIT_COUNTS`。先核对服务器脚本版本、运行 `bash -n`，再按实际 `new_split` 设置：

```bash
SPLIT_MANIFEST="$new_split/final_manifest.tsv" DATA_ROOT="$HOME/Data_V2" BASE_REPORT="$HOME/Code/pipeline_reports/DATA_V2_50X4_CONFIDENCE" ALLOW_VARIABLE_SPLIT_COUNTS=1 bash ~/Code/predict_protenix/Version_3/data_v1_50x4_confidence/start_data_v1_50x4_confidence_8gpu.sh
```

等待所有入选 PDB 的 prep/pred 审计完成，并保存本次 `pred_runs/<RUN_ID>` 路径。

## 5. 生成新版 PT，单独保存 >30 Å

```bash
python ~/Code_Flow_matching/scripts/build_refinement_pt.py --prediction-root ~/Data_V2 --native-root ~/pdb_data --rnafm-root ~/Data_FM/RNA_FM_embeddings --output-root ~/Data_PT_V2 --high-rmsd-root ~/Data_PT_V2_RMSD_GT30 --exclude-pdb-file ~/Code_Flow_matching/config/refinement_excluded_pdb_ids_v2.tsv --log-dir ~/Code/pipeline_reports/PT_V2 --run-name write_v2_rmsd30_20260924 --ccd-components-file ~/protenix_data/common/components.cif --max-pre-refinement-rmsd 30
```

V2 排除名单保留原有数据质量排除项，移除了仅因构象 RMSD 极高而整 PDB 排除的 `9ZC9`；它的 >30 Å 样本应进入单独的高 RMSD PT 目录。日志会记录原策略中的 split 和本次实际 split。主训练 PT 根目录不会包含 >30 Å 样本。检查 `summary.json` 中 `issues`、`failed_samples`、`rmsd_filtered_samples` 和 `high_rmsd_saved_samples`，后两项应一致。重跑前应使用新的 run name；只有明确需要重建样本时才加 `--overwrite`。

## 6. 分析 ranking_score，再决定阈值

```bash
python ~/Code_Flow_matching/scripts/analyze_ranking_vs_rmsd.py --pt-run-dir ~/Code/pipeline_reports/PT_V2/write_v2_rmsd30_20260924 --output-dir ~/Code/pipeline_reports/PT_RANKING_RMSD_V2_20260924 --rmsd-threshold 30
```

脚本输出样本级表、PDB 级表、各 split 的低分识别 >30 Å 的 AUC 和阈值曲线。阈值只从 train/val 选择；test 只用于最终报告。如果分数区分能力不够，不应仅靠 ranking_score 排除 >30 Å 样本。

旧版探索分析也可对 `~/Data_PT_V1/logs/write_v1_rmsd30_20260916` 运行同一脚本，另加 `--prediction-root ~/Data_V1` 从 Protenix summary confidence JSON 补旧日志缺失的 `ranking_score`。旧版结果不能替代新版 train/val 阈值选择。

## 7. 生成新版逐 PDB 状态表

用第 1 步相同的审计脚本，将 `--split-report`、`--prediction-root`、`--pt-root`、`--high-pt-root`、三个 `--pred-audit` 和 `--pt-log-dir` 改为第 2～5 步对应的新版路径，输出到 `~/Code/pipeline_reports/PDB_LIFECYCLE_V2_<日期>/`。每行显示是否入选及原因、prep/pred 审计、PT 数量、>30 Å 数量、单独保存数量和 PT 策略排除原因。

## 8. 训练时对 >50 nt 上采样

训练配置使用 `Data_PT_V2`。在 `datamodule_args` 下设置 `train_length_table: <new_split>/selection_audit.tsv`、`long_rna_threshold_nt: 50` 和 `long_rna_oversample_factor: 2.0`。采样器只作用于 train；每个 epoch 仍抽取与训练 PT 数量相同的样本数，>50 nt 单个 PT 相对于 ≤50 nt 单个 PT 的抽取权重为 2 倍。验证和测试不采样。实际倍数应在训练前依据新版长度分布、显存和验证集表现确定。

模型选择只看验证集。新测试集和 PT 完成后，使用已由验证集确定的 checkpoint 运行 `evaluate_refinement.py --split test --data-dir ~/Data_PT_V2`，结果写入新的 `~/Code/pipeline_reports` 类别目录，不覆盖旧 `evaluation`。
