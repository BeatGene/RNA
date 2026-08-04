# 实验纯 RNA PDBx/mmCIF 数据管线

这个管线同时完成三件事：

1. 按旧清单审计从之前服务器迁移的 1979 个 CIF 是否存在、能否严格解析、PDB ID
   是否一致、坐标列是否完整；
2. 只按当前 2241 个“实验纯 RNA”ID 清单增量下载缺失或损坏的 CIF；
3. 生成结构级清单和 RNA 链级序列记录。

旧清单独有的 5 个 ID（`176D`、`3OK2`、`3OK4`、`5EME`、`5EMF`）只参与
迁移文件完整性审计，不会被当作当前下载目标，也不会被删除。

## 输出

默认输出到 `<scan-root>/pipeline_reports/`：

- `pdb_cif_report.xlsx`
  - `结构清单`：每个 PDB 一行，含文件状态、SHA-256、实验方法、polymer 类型等；
  - `RNA链序列`：每条 RNA 链一行，含 entity、chain、原始/标准序列；
  - `汇总`：旧 1979 个文件和当前 2241 个目标的完成统计。
- `pdb_cif_manifest.csv`：完整、机器可读的结构清单；
- `rna_chain_sequences.csv`：完整序列，不受 Excel 单元格长度限制；
- `summary.json`：最简运行摘要；
- `run_events.jsonl`：逐个 ID 追加的断点/审计日志；
- `quarantine/`：仅在修复同名损坏文件时保存损坏原件的副本。

绿色表示文件有效，黄色表示缺失，红色表示损坏。`PURE_RNA_POLYMERS` 描述 CIF
自身声明的 polymer 组成；小分子配体、离子和水不属于 polymer，不影响该字段。

## 在之后的实验室服务器上运行

把整个 `Code/Download_PDB_RAW` 目录传到服务器，保证脚本和两份 Excel 的相对目录
不变。然后：

```bash
cd ~/Download_PDB_RAW
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements_pdb_pipeline.txt
```

先只校验，不下载：

```bash
python3 pdb_cif_pipeline.py audit --scan-root ~/pdb_data
```

如果旧 CIF 实际位于 `~/pdb_data/01_Pure_RNA`，脚本会递归找到它们。确认审计报告
后，同步缺失/损坏的当前目标：

```bash
python3 pdb_cif_pipeline.py sync \
  --scan-root ~/pdb_data \
  --download-dir ~/pdb_data/01_Pure_RNA
```

如果服务器上的目标子目录名称不同，只需要修改 `--download-dir`。运行中断后直接
执行同一条命令即可；已通过校验的文件不会重复下载。下载先写入隐藏的 `.part`
临时文件，通过严格解析后才原子落盘。脚本不会删除任何现有 CIF；同名损坏文件在
替换前会复制到 `pipeline_reports/quarantine/`。

命令最终返回码为 `0` 表示当前 2241 个目标全部有效且旧 1979 个文件全部完整；
返回码 `2` 表示报告中仍有缺失、损坏或下载失败项。

## 字段口径

- `FILE_STATUS`：`VALID`、`MISSING` 或 `INVALID`；
- `SYNC_STATUS`：已有文件有效、下载后有效、下载失败等本次动作结果；
- `ELIGIBILITY_STATUS`：当前清单中的文件是否也由 CIF 元数据确认为纯 RNA polymer；
- `EXPERIMENT_METHOD`：直接读取 `_exptl.method`；
- `SEQUENCE_REPORTED`：`_entity_poly.pdbx_seq_one_letter_code`，保留括号中的修饰残基；
- `SEQUENCE_CANONICAL`：`_entity_poly.pdbx_seq_one_letter_code_can`；
- `CHAIN_ID`：`_entity_poly.pdbx_strand_id`；
- `SHA256`：文件内容指纹，可用于后续搬运前后逐字节比对。

完整性校验包括：CIF 语法检查、空/重复项检查、单 data block、`_entry.id` 匹配、
实验方法与 polymer 元数据存在、`_atom_site` 的 ID/X/Y/Z 列等长且所有坐标为有限
数。当前目标还会另行核对 polymer 是否全部为 RNA，并检查 RNA entity 序列。这样
不会把完整但已不符合当前纯 RNA 口径的旧文件误报为损坏。它能可靠发现空文件、
截断、语法损坏、错名文件和关键数据缺失；SHA-256 则用于今后的搬运一致性校验。
