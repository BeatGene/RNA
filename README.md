# Code

## Download_PDB_RAW
下载原始的cif文件

### alignment_pdb_ids_not_in_experimental_pure_rna.xlsx

5个原来脚本判断为纯RNA实际不是的PDB_id

### experimental_pure_rna_pdb_ids.xlsx

表1:2241个纯RNA的PDB_id 表2:RNARefine文章中涉及到的PDB_id 实际纯RNA的PDB_id已经全部在表1中了

### rna_entity_metadata.xlsx

2241+77个cif文件的metadata 由build_rna_entity_metadata.py脚本生成（包括纯的和杂牌的）

### requirements_pdb_pipeline.txt

脚本pdb_cif_pipeline.py的环境要求

### download_out_cif.py

从experimental_pure_rna_pdb_ids.xlsx表2中下载OUT的cif文件(杂牌)

## pipeline_reports

**注意:从New_Data_pipeline_reports改名而来 与实验室服务器上保持一致**

### DATA_SPLIT

划分数据集的log日志

#### V0

第零版 ~/Data

从2241个cif文件中 

1.先按时间划分训练集 验证集与测试集 （PDB单位）

2.然后测试集对验证集和训练集去同源再内部去冗余。序列一致性及双向覆盖率的阈值均为 80% 只要还有一条链 就保留PDB(更早期是命中就移除整个PDB)

#### V1

第一版 ~/Data_V1

从2241个cif文件中 

1.先按时间划分训练集 验证集与测试集 （PDB单位）

2.只选择仅包含单链RNA的cif文件 同时要有ranking_score_1 RMSD记录的（PDB单位）

3.其中对于训练集，ranking_score_1与native cif的foldbench RMSD大于15Å的都不要(PDB单位)

4.然后测试集对验证集和训练集去同源再内部去冗余。序列一致性及双向覆盖率的阈值均为 80%

#### V2

第二版 ~/Data_V2

从2241个cif文件中 

1.先按时间划分训练集 验证集与测试集 （PDB单位）

2.只选择仅包含单链RNA的cif文件 （PDB单位）

3.然后测试集对验证集和训练集去同源再内部去冗余。序列一致性及双向覆盖率的阈值均为 80%

### DATA_V2_FOLDBENCH_TEST_OVERRIDE_20260930T081218Z

把验证集中foldbench的移动到测试集

### DATA_DECOY_GENERATION

生成包含置信度decoy的log日志
#### V1

~/Data_V1
生成包含置信度的decoy

#### V2

~/Data_V1
补充生成41个包含置信度的decoy 其余符号链接到V1

### RNA_FM

生成RNA_FM的log日志

### DATA_PT_GENERATION

pt生成日志

#### V2

~/Data_PT_V2

去除掉大于30Å的.pt文件 但是保存在~/Data_PT_V2_RMSD_GT30

只生成41个新的cif的.pt文件

### FOLDBENCH

### DECOYS

最开始进来的时候审计

## RNA_FM_pipeline

用于生成RNA_FM


# Code_Audit

保存了代码审查的文件

# Data_PT_V1

实验室服务器上保存的基本单位是.pt文件，本地保存的是两次冒烟、一次真正跑的log记录

# Json_data

 保存了foldbench计算5*5RMSD的结果，用来评估protenix预测的RNA构象的质量；保存了配对mapping情况

# PPT

保存了历次汇报的PPT

# README.md

本文件记录了本地以及github仓库中的文件情况

# SERVER.md

记录了实验室服务器上的文件情况

# wb.txt

保存了wandb的API KEY