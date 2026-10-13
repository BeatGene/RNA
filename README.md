# Code
## Download_PDB_RAW 
alignment_pdb_ids_not_in_experimental_pure_rna.xlsx 5个原来脚本判断为纯RNA实际不是的PDB_id 

experimental_pure_rna_pdb_ids.xlsx                  表1:2241个纯RNA的PDB_id 表2:RNARefine文章中涉及到的PDB_id 实际纯RNA的PDB_id已经全部在表1中了

rna_entity_metadata.xlsx                            2241+77个cif文件的metadata 由build_rna_entity_metadata.py脚本生成（包括纯的和杂牌的）

requirements_pdb_pipeline.txt                       脚本pdb_cif_pipeline.py的环境要求

download_out_cif.py                                 从experimental_pure_rna_pdb_ids.xlsx表2中下载OUT的cif文件(杂牌)

## pipeline_reports
DATA_DECOY_GENERATION 生成包含置信度decoy的log日志

DATA_PT_GENERATION  pt生成日志

DATA_SPLIT 划分数据集的log日志

DATA_V2_FOLDBENCH_TEST_OVERRIDE_20260930T081218Z 把验证集中foldbench的移动到测试集



RNA_FM 生成RNA_FM的log日志

## predict_protenix 
生成decoy的脚本

## RNA_FM_pipeline
用于生成RNA_FM的脚本

## Split_RNA_dataset
用于划分数据集的脚本

# Code_Audit
保存了代码审查的文件

# Code_Flow_matching

# Json_data
保存了foldbench计算5*5RMSD的结果，用来评估protenix预测的RNA构象的质量

# PPT
保存了历次汇报的PPT

# README.md
本文件记录了本地以及github仓库中的文件情况

# SERVER.md
记录了实验室服务器上的文件情况

# wb.txt
保存了wandb的API KEY