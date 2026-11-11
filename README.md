# Code
## Download_PDB_RAW 
alignment_pdb_ids_not_in_experimental_pure_rna.xlsx 5个原来脚本判断为纯RNA实际不是的PDB_id 

check_cifs_against_rcsb.py                          检查服务器上2241+77cif文件

download_out_cif.py                                 从experimental_pure_rna_pdb_ids.xlsx表2中下载OUT的cif文件(杂牌)

experimental_pure_rna_pdb_ids.xlsx                  表1:2241个纯RNA的PDB_id 表2:RNARefine文章中涉及到的PDB_id 实际纯RNA的PDB_id已经全部在表1中了

pdb_cif_pipeline.py                                 下载cif文件的主程序脚本

requirements_pdb_pipeline.txt                       脚本pdb_cif_pipeline.py的环境要求

rna_entity_metadata.xlsx                            2241+77个cif文件的metadata 由build_rna_entity_metadata.py脚本生成（包括纯的和杂牌的）

## pipeline_reports
日志文件

DATA_DECOY_GENERATION 生成包含置信度decoy的log日志

DATA_PT_GENERATION  pt生成日志

DATA_SPLIT 划分数据集的log日志

DATA_V2_FOLDBENCH_TEST_OVERRIDE_20260930T081218Z 把验证集中foldbench的移动到测试集

RNA_FM 生成RNA_FM的log日志

## predict_protenix

### MSA_cached_input
cached_rna_prep.py 以唯一RNA序列为单位准备MSA。优先复用已有有效MSA、当前数据库集合的缓存；可选使用官方预计算MSA；其余序列才进行新搜索。

### qc
check_simple_json.py  原始JSON是否与实验CIF、服务器安装的Protenix转换器结果一致，包括序列、拷贝数、修饰、配体和共价键

check_prep.py         raw/final JSON内容是否一致、A3M全部记录是否可解析、query是否匹配、路径和链接是否有效

deep_prep_audit.py    序列级prep检查，并与已有链序列清单交叉核对；默认范围偏向旧数据集合

qc/compare_cif_ids.py 对比实验CIF与预测CIF的链ID、残基编号和序列，方便定位编号差异

### share_predict_core
stage2_decoy_pipeline.py  总调度程序，提供audit、make-json、prep、pred、preflight子命令。负责检查已有产物、寻找缺失任务、分配GPU、记录日志、生成审计报告。

resident_protenix_pred.py 实际执行推理的GPU worker。在一个进程中复用已加载的模型，连续处理多个目标和seed。
## RNA_FM_pipeline
用于生成RNA_FM的脚本

## Split_RNA_dataset
用于划分数据集的脚本

# Code_Audit
保存了代码审查的文件

[//]: # (ToDo)
# Code_Flow_matching
Code_Flow_matching/config 模型yaml配置文件

Code_Flow_matching/etflow 模型本体

Code_Flow_matching/evaluation 

Code_Flow_matching/scripts 待检查 

# PPT
保存了历次汇报的PPT

# AGENT.md
vibe coding 需要遵守的规则

# README.md
本文件记录了本地以及github仓库中的文件情况

# SERVER.md
记录了实验室服务器上的文件情况

# wb.txt
保存了wandb的API KEY