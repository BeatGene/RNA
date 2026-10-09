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
# Code_Audit

保存了代码审查的文件

# Data

与实验室服务器上的Data含义不同 保存了foldbench计算5*5RMSD的结果，用来评估protenix预测的RNA构象的质量；保存了配对mapping情况

# Data_PT_V1

保存的基本单位是.pt文件，本地保存的是两次冒烟一次真正跑的log记录

# PPT

保存了历次汇报的PPT

# wb.txt

保存了wandb的API KEY