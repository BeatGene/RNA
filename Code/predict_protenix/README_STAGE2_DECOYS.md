# 第二阶段：Protenix RNA decoy 管线

> 历史说明：本文记录旧的 4 seeds x 50 samples decoy 任务。当前要求的
> FoldBench-style 第一阶段（prep + 5 seeds x 5 samples pred）请使用
> `README_FOLDBENCH_STAGE1.md` 和 `run_foldbench_stage1.sh`，不要执行本文的
> 旧预测命令。

## 目标与判定标准

当前目标来自第一阶段 `pdb_cif_manifest.csv` 中
`CURRENT_TARGET=True` 的 2241 个 PDB，而不是直接扫描目录中的 2246 个 CIF。
另外 5 个 legacy-only CIF 不进入本阶段。

每个目标的最终完成条件：

1. 原始 Protenix JSON 存在且能解析；
2. `*-final-updated.json` 存在，每个 RNA 序列均有非空 RNA MSA；
3. seed 42、43、44、45 各有 50 个主预测 CIF；
4. 主 CIF 的 sample 编号必须恰好为 0–49 且文件可读取；
5. 总计 200 个有效 decoy。

`*_wounresol.cif` 是额外的去未解析原子版本，会单独统计，不能混入 50 个
主 CIF。程序不会仅凭 Protenix 的进程退出码判定成功，而会检查实际产物。

## 文件

- `stage2_decoy_pipeline.py`：审计、补 JSON、补 prep、补 pred；
- `requirements_stage2_decoys.txt`：报告和 CIF 完整校验依赖；
- `tests/test_stage2_decoy_pipeline.py`：不需要 GPU 的本地测试。

脚本只补缺失或未通过验收的任务，不删除现有结果。旧
`*-final-updated.json` 中失效的旧服务器绝对路径会标记为
`COMPLETE_REBASABLE`。预测前程序在报告目录生成运行时 JSON 并修正路径，
不会覆盖原文件。

## 第一步：传到新服务器

在本地 PowerShell、项目根目录执行：

```powershell
scp -P 30063 Code/predict_protenix/stage2_decoy_pipeline.py `
  tinghao.xia@dubhe.lglab.ac.cn:~/Code/predict_protenix/

scp -P 30063 Code/predict_protenix/requirements_stage2_decoys.txt `
  tinghao.xia@dubhe.lglab.ac.cn:~/Code/predict_protenix/
```

若远端目录还不存在，先登录服务器运行：

```bash
mkdir -p ~/Code/predict_protenix ~/Code/pipeline_reports/DECOYS
```

## 第二步：先做只读审计

该步骤不需要 Protenix、数据库或 GPU：

```bash
conda activate rna_pdb
python -m pip install -r ~/Code/predict_protenix/requirements_stage2_decoys.txt

nohup python ~/Code/predict_protenix/stage2_decoy_pipeline.py audit \
  --cif-validation quick \
  > ~/Code/pipeline_reports/DECOYS/audit_console.log 2>&1 &
echo $!
```

查看进度和结果：

```bash
tail -f ~/Code/pipeline_reports/DECOYS/audit_console.log
cat ~/Code/pipeline_reports/DECOYS/summary.json
```

报告包括：

- `decoy_report.xlsx`：易读工作簿；
- `decoy_manifest.csv`：每个 PDB 的 JSON、prep、4 个 seed 和总状态；
- `decoy_seed_manifest.csv`：每个 PDB × seed 的 50 个样本详情；
- `chain_id_mapping.csv`：Protenix 链 A/B/... 到原链 ID 的序列映射；
- `summary.json`：机器可读汇总；
- `run_events.jsonl`：后续补算时每次子进程的参数、时间、GPU、退出码和日志。

建议先把 `summary.json` 和审计日志内容发回本地确认，再安装和补算。

## Protenix 环境建议

服务器已经可直接访问 Docker daemon，不需要再启动一个嵌套的
`dockerd`。建议使用官方运行时镜像：

```text
ai4s-share-public-cn-beijing.cr.volces.com/release/protenix:1.0.0.4
```

该镜像包含 PyTorch、HMMER、Kalign、CUTLASS 等运行依赖，但不包含
Protenix 源码、模型权重和数据库。应先检查公共存储及已有镜像，避免重复
下载。

服务器上执行以下只读检查：

```bash
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' \
  | grep -Ei 'protenix|pytorch|cuda' || true

findmnt -rn -o TARGET,SOURCE,FSTYPE | sort

for root in /share /data /mnt /storage9920/share /storage9920/public \
            /remote-home/share; do
  if [ -d "$root" ]; then
    echo "===== $root ====="
    find "$root" -maxdepth 7 -type f \
      \( -name 'protenix_base_default_v1.0.0.pt' \
      -o -name 'components.cif' \
      -o -name 'pdb_seqres_2022_09_28.fasta' \
      -o -name 'nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta' \
      -o -name 'rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta' \
      -o -name 'rnacentral_active_seq_id_90_cov_80_linclust.fasta' \) \
      -printf '%s %p\n' 2>/dev/null
  fi
done
```

至少需要下面这些内容位于同一个 `PROTENIX_ROOT_DIR`，或在执行 prep
时显式传入数据库路径：

```text
common/components.cif
common/components.cif.rdkit_mol.pkl
common/obsolete_release_date.csv
common/clusters-by-entity-40.txt
checkpoint/protenix_base_default_v1.0.0.pt
search_database/pdb_seqres_2022_09_28.fasta
search_database/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
search_database/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
search_database/rnacentral_active_seq_id_90_cov_80_linclust.fasta
```

确定公共路径后，再固定 Docker 镜像、Protenix 源码 commit 和数据挂载。
不要先盲目下载全量训练数据库；本任务只需要 inference common、
checkpoint 和 search databases。

## 环境预检

在最终 Protenix 运行环境内部执行。例如公共数据根目录为
`/shared/protenix_data`：

```bash
export PROTENIX_ROOT_DIR=/shared/protenix_data

python ~/Code/predict_protenix/stage2_decoy_pipeline.py preflight
```

若数据库不在标准目录，可显式指定：

```bash
python ~/Code/predict_protenix/stage2_decoy_pipeline.py preflight \
  --seqres-database /path/pdb_seqres_2022_09_28.fasta \
  --ntrna-database /path/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta \
  --rfam-database /path/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta \
  --rnacentral-database /path/rnacentral_active_seq_id_90_cov_80_linclust.fasta
```

## 增量补算顺序

只有预检通过后才依次运行。

### 1. 补 267 个新目标的原始 JSON

```bash
python ~/Code/predict_protenix/stage2_decoy_pipeline.py make-json --workers 4
```

### 2. 补 prep

`prep` 是 CPU/HMMER 任务，不使用 GPU。默认 4 个并行任务，每个
`nhmmer` 最多 8 CPU：

```bash
python ~/Code/predict_protenix/stage2_decoy_pipeline.py prep \
  --workers 4 --nhmmer-cpus 8
```

若数据库使用非标准路径，把预检中的四个数据库参数原样附加。

### 3. 补预测

旧服务器脚本参数已保留：模型
`protenix_base_default_v1.0.0`、bf16、10 cycles、200 steps、每个 seed
50 samples、seed 42–45、RNA MSA 开启、template 关闭。

当前 GPU 0–3 已占用约 131 GiB 显存，先只使用空闲的 4–7：

```bash
nohup python ~/Code/predict_protenix/stage2_decoy_pipeline.py pred \
  --gpus 4,5,6,7 \
  > ~/Code/pipeline_reports/DECOYS/pred_console.log 2>&1 &
echo $! > ~/Code/pipeline_reports/DECOYS/pred.pid
```

每张 GPU 同时只运行一个 seed 任务。可用以下命令观察：

```bash
tail -f ~/Code/pipeline_reports/DECOYS/pred_console.log
nvidia-smi
```

中断后重复同一条命令即可：已完整的 seed 会跳过，不完整的 seed 会重跑并
在结束后重新验收。

## 链 ID 和残基 ID

当前 Protenix 会根据 JSON 顺序重新生成链 `A/B/...`，并把 polymer
残基按 `1..L` 编号；输入 JSON 中即使附带原始 `label_asym_id`，当前
特征构建代码也不会用它还原输出编号。因此“让模型原生输出原编号”不可行。

可行方案是预测后重写 author 编号：

1. 通过序列把预测链映射回原始 `_atom_site.auth_asym_id`；
2. 通过原始 `_pdbx_poly_seq_scheme` 将序列位置映射到
   `auth_seq_id` 和 insertion code；
3. 保留 Protenix 的 label 编号，另存带原 author 编号的 CIF。

`chain_id_mapping.csv` 先记录第 1 步。不同序列的链可精确匹配；相同序列
副本只凭序列无法区分，会标为 `IDENTICAL_SEQUENCE_ORDER_ASSUMED`。若后续
对齐要求区分这些对称链，应在坐标层面枚举链置换并选择最小 RMSD，不能假装
它是唯一映射。
