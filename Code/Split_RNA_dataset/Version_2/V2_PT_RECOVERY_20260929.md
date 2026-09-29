# V2 PT 补齐与逐 PDB 核对

本流程保留已完成的 V2 划分。`9R82` 因 7587-token 复合物在 140 GiB GPU 上预测 OOM，保留空预测目录，跳过 PT。10 个已入选但放弃 prep 的 PDB 同样跳过；4 个未入选的 3DG* 仅保留在状态名单中。V2 的 PT 策略排除清单覆盖 56 个已入选 PDB，不把它们当成 PT 失败。已复用的 964 个预测目录不再预测。

新增 train 的 PT 首轮运行 `step1_new_train_20260929T073447Z` 在 40×200 个预测中生成 5620 个正常 PT、1180 个单独保存的高 RMSD PT。其余 6 个 PDB 各有 200 个 PT 失败：`1QZA`、`1QZB`、`2Z9Q` 的 native 各仅有约 75 个 RNA 原子，`5K8H` 的 7 个残基中只有 2 个有坐标；`2JJA`、`2XC6` 涉及不能证明唯一 A/C/G/U 身份的修饰残基。它们仍属于 V2 train，但不作为训练 PT 使用；原因及逐样本失败证据保存在 `pt_unusable_new_train_v2.tsv` 和原运行的 `issues.tsv`。不得通过降低监督比例阈值或强行指定未知碱基来伪造可用 PT。

## 目录与产物

- 正常 PT：`~/Data_PT_V2/<split>/<pdb>/seed_<seed>/sample_<index>.pt`。
- RMSD＞30 Å 的 PT：`~/Data_PT_V2_RMSD_GT30/...`，训练和验证的数据加载器只读取前一个根目录。
- 各阶段日志：`~/Code/pipeline_reports/PT_V2_REUSE_*`、`PT_V2/step1_new_train_*`、`PT_V2/step2_retained_*`、`PT_V2_AUDIT_*`。
- 旧 `Data_PT_V1` 保留；V2 复用旧正常 PT 时创建逐文件硬链接，不创建 PDB 目录符号链接，也不覆盖已有目标。

## 先后顺序

1. 上传 `V2_PT_CODE_20260929.zip` 到 `~/Code_Flow_matching/`，上传 `V2_PT_WORKFLOW_20260929.zip` 到 `~/Code/Split_RNA_dataset/Version_2/`。分别在这两个目录执行 `python3 -m zipfile -e <包名> .`。运行 `bash -n` 检查所有新 shell 脚本。
2. 运行 `bash ~/Code/Split_RNA_dataset/Version_2/run_v2_pt_reuse_plan.sh dry-run`。核对 `summary.json`：988 个 V1、1015 个 V2、964 个保留、40 个新增 train PT 目标、908 个旧目标可复用、11 个已入选上游跳过、56 个 V2 PT 策略排除；`blockers=0`。保存此 dry-run 报告路径。
3. 新增 train 的 PT 首轮已经运行。先用 `validate_v2_new_train_pt.py` 核对全部 8000 个候选：34 个 PDB 各有 200 个保存的正常或高 RMSD PT；6 个有稀疏 native 坐标或修饰残基歧义的 PDB 各有 200 条预期失败记录，且无 PT 文件。验证器发现任何额外失败、重复样本、缺失文件或错误原因变化都会停止。
4. 运行 `bash ~/Code/Split_RNA_dataset/Version_2/run_v2_pt_reuse_plan.sh execute <dry-run报告目录> <第3步PT报告目录>`。脚本先自动验证第 3 步的实际运行结果，再将 V2 保留且符合 PT 策略的旧主 PT 逐文件硬链接到新 PT 根目录。检查 `created_hardlinks+already_hardlinked=old_main_pt_files`、`blockers=0`；保存 execute 报告路径。
5. 执行 `bash ~/Code/Split_RNA_dataset/Version_2/run_v2_pt_retained.sh <execute报告目录> <第3步PT报告目录>`。此步处理 908 个保留旧目标；已有正常 PT 经版本检查后跳过，旧版因 RMSD＞30 Å 未保存的候选重新构建至单独目录，旧版 `9ZC9` 的 200 个预测也纳入。若有问题，报告仍会列出逐样本 `issues.tsv`；不要删除目录或用 `--overwrite`。
6. 执行 `bash ~/Code/Split_RNA_dataset/Version_2/run_v2_pt_final_audit.sh <第3步PT报告目录> <第5步PT报告目录>`。输出 `per_pdb.tsv`、`missing_samples.tsv`、`old_gt30_recovery.tsv`、`summary.json`、`ranking_vs_rmsd/` 和 `lifecycle/pdb_lifecycle.tsv`。逐 PDB 表检查有 200 个预测的 PDB 是否得到合计 200 个正常或高 RMSD PT；6 个已确认 PT 不可用的 PDB 单独标明稀疏 native 坐标或修饰残基映射歧义，其余缺失样本按原因分类。`lifecycle/` 涵盖全部 2246 个源 PDB，并保留未划分原因。

`Data_PT_V2` 与 `Data_PT_V1` 的硬链接共享底层文件；生成器以临时文件原子替换 V2 路径，未使用 `--overwrite`，不会原地改写 V1 文件。之后若要修改 PT 文件内容，应先将对应硬链接复制成独立文件。旧版 `filtered_samples.tsv` 中 3168 个候选属于整个 V1 选集；最终 `old_gt30_recovery.tsv` 只统计 V2 中保留且符合 PT 策略的交集，不应要求全部 3168 个进入 V2。

ranking_score 分析以新生成的两个 PT 报告合并后的样本为准；阈值只能用 train/val 决定，test 仅用于最终报告。分析不自动按分数删除 PT。

## 查看进度

各阶段报告目录下有 `run.log`。例如运行新 train PT 时，脚本先打印 `PT run directory`，然后可在另一个 SSH 会话执行 `tail -f <该目录>/run.log`。保留旧目标的 PT 生成、复用计划及逐样本审计同理。最终审计的三个子步骤分别在 `<审计目录>/run.log`、`<审计目录>/lifecycle/run.log`、`<审计目录>/ranking_vs_rmsd/run.log`。

`PROGRESS` 行包含进度条、百分比、已处理数量和大致剩余时间。PT 生成每处理完一个 PDB 就更新，并在长样本处理中每 60 秒写一次心跳；`active` 显示当前 PDB/seed/sample。`eta_approx` 仅根据已完成样本的平均速度估算，长链样本可能使它大幅变化。若有样本在找到预测之后、进入逐样本循环之前发生 PDB 级错误，样本进度可能达不到 100%；此时检查 `issues.tsv` 和 `summary.json`，不要把百分比当成成功率。
