# 直接核对 Simple JSON 与 RNA-FM embedding

`check_rnafm_json_alignment.py` 是独立脚本，只需要 Python >=3.9 和支持
`torch.load(..., weights_only=True)` 的 PyTorch（当时的 `rna_fm` 环境即可）。
不需要 GPU、RNA-FM 包、模型权重或旧的审计表。只读取原始 JSON 和 PT，
检查结果写入新报告目录，不修改输入、不重新生成 embedding。

## 上传和运行

将 **一个脚本** 上传到服务器：

```bash
scp Code/RNA_FM_pipeline/check_rnafm_json_alignment.py \
  tinghao.xia@YOUR_SSH_HOST:/storage9920/home/tinghao.xia/
```

将 `YOUR_SSH_HOST` 换成平时登录的地址；如使用自定义 SSH 端口，给 scp
增加 `-P 端口号`。上传命令从本地 RNA 项目根目录执行。

服务器上运行：

```bash
conda activate rna_fm
python /storage9920/home/tinghao.xia/check_rnafm_json_alignment.py
```

默认读取：

```text
/storage9920/home/tinghao.xia/Json_data/Simple_json/<pdb>.json
/storage9920/home/tinghao.xia/Data_FM/RNA_FM_embeddings/<PDB>/rnafm_t12_residue_embeddings.pt
```

默认要求恰好 2241 个目标。读取名字形如 `9zi7.json` 的原始文件，忽略
`*-final-updated.json` 等 prep 文件，并排除当时的五个 legacy-only ID：
`176D、3OK2、3OK4、5EME、5EMF`。大小写不同但 ID 相同的重复路径会报错。
输出缺失、额外目标、重复目录和额外 PT 文件均会使总检查失败。

可选：使用指定报告目录（必须尚不存在，且位于两个输入目录之外）：

```bash
python /storage9920/home/tinghao.xia/check_rnafm_json_alignment.py \
  --report-dir /storage9920/home/tinghao.xia/RNA_FM_JSON_CHECK_manual_01
```

## 检查内容

1. JSON task 名称与 PDB ID 一致，按 RNA 条目的 `count` 展开链；
2. PT 内序列、链顺序、序列哈希、JSON 条目索引与拷贝索引一致；
3. 每条链的 `chain_offsets` 对应准确的切片边界；
4. `residue_embedding` 为 `[总残基数, 640]`，每个残基都有一行有限且非全零向量；
5. `residue_chain_index` 和 `residue_index_in_chain` 与每个残基位置一致；
6. token ID、非 ACGU 掩码、UNK 掩码与输入序列一致；N 保留为 N，X/I/T 为 UNK；
7. 模型与提取层等元数据符合原来的 `rna_fm_t12` 第 12 层格式。

序列比较采用与原生成器相同的去空白、转大写处理，不改变碱基、不截断序列。
原始字符串需要格式规范化时会写入 WARNINGS。非 RNA 实体不在 RNA 残基检查范围内，
会明确记录警告。原始链 ID 会展示在报告中，但没有 CIF 输入，因此不独立证明
实验链/坐标对应关系，也不验证修饰、配体或共价键的化学编码。

本检查证明现有文件的序列、位置索引和向量行数对应，并检查数值有效性；
不会重新运行模型，因此不独立重算确认每一行向量是否等于 RNA-FM 的实际前向结果。

## 查看结果

程序会打印 `REPORT_DIR=...`，默认目录位于服务器家目录，名字以
`RNA_FM_JSON_CHECK_` 开头。里面包含：

- `summary.json`：总体结论、缺失/额外目标、通过/失败数量、总链数与残基数；
- `per_pdb.tsv`：逐 PDB 比较结果；
- `per_chain.tsv`：逐链长度、切片位置、embedding 行数及序列对应情况；
- `issues.tsv`：具体错误；
- `check.log`：进度与最终结果。

最终成功需要同时满足：

```text
status = PASS
all_targets_passed = true
target_json_id_count = 2241
embedding_id_count = 2241
pass_pdbs = 2241
failed_pdbs = 0
issue_count = 0
```

`summary.json` 的最终总体结论为准；单独的链行 PASS 不覆盖其所属 PDB 的
其他元数据错误。若输入与当时完全相同，预期还会看到 4304 条链、396773 个残基；
这两个值由本次当前 JSON 重新统计，不拿历史数值替代检查。

退出码：0 全部通过；1 存在不匹配；2 参数、依赖或致命错误；130 手动中断。
运行中会持续更新总体汇总；中断后的报告不能视作完整通过。

## 断开连接后恢复，并转为后台运行

先重新登录服务器，确认原进程是否还在运行：

```bash
pgrep -af '[c]heck_rnafm_json_alignment.py'
```

若仍有原检查进程，不要向同一个报告目录重复启动；可以先等待其结束。
脚本新版有报告目录锁，但最初的 1.0 版没有锁，因此不能靠新版阻止旧进程。

原进程已退出后，重新上传新版脚本。然后激活环境，将下面报告路径改为
之前屏幕打印的 `REPORT_DIR`（也可以查看家目录下 `RNA_FM_JSON_CHECK_*` 目录）：

```bash
conda activate rna_fm
RNAFM_REPORT=/storage9920/home/tinghao.xia/RNA_FM_JSON_CHECK_实际时间戳

nohup "$CONDA_PREFIX/bin/python" -u \
  /storage9920/home/tinghao.xia/check_rnafm_json_alignment.py \
  --resume --report-dir "$RNAFM_REPORT" \
  >> "$RNAFM_REPORT/background.log" 2>&1 < /dev/null &
echo $! > "$RNAFM_REPORT/background.pid"
```

这是后台任务，SSH 断开后仍会运行；服务器/容器重启仍会终止任务，届时再次
执行相同命令即可恢复。若最初使用了自定义输入路径、数量或排除名单，恢复时
必须加上完全相同的参数。

恢复策略：

- 已有的 `summary.json`、TSV 报告先备份到 `previous_runs/<时间戳>/`；
- 新版每检查完一个 PDB，就原子保存 `checkpoints/<PDB>.json`；
- 只复用同一脚本版本/内容和 PyTorch 环境下检查通过，且 JSON、PT 文件的
  路径、大小、时间戳、inode 等指纹没有变化的结果；新加额外 PT 也会使缓存失效；
- 失败、未完成、文件已改变的 PDB 会重新检查；脚本内容或 PyTorch 版本变化时
  也会重新核验；
- **最初 1.0 版没有保存 PT 指纹，因此第一次用新版恢复旧报告，会备份旧结果并
  重新核验一遍。不能仅依据旧 TSV 的 PASS，就假定文件至今未变化。**

查看后台进度：

```bash
tail -n 20 "$RNAFM_REPORT/background.log"
python -m json.tool "$RNAFM_REPORT/summary.json"
```

`reused_pdbs` 是本轮复用数量，`rechecked_pdbs` 是本轮实际检查数量；
`per_pdb.tsv` 中的 `CHECK_ACTION` 分别记录 `REUSED_PASS` 或 `CHECKED`。
最终仍以 `status=PASS`、`all_targets_passed=true` 为准。

只有确实检查另一套数据时才改 `--expected-count`；`--exclude` 单独使用可以
清空默认排除列表。不要为使报告通过而随意调整预期数量或排除名单。

本地测试（需要 PyTorch）：

```bash
python -m unittest discover -s Code/RNA_FM_pipeline/tests -p 'test_check_rnafm_json_alignment.py' -v
```
