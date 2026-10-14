# ~/Code
~/Code/FoldBench FoldBench代码

~/Code/pipeline_reports日志所在地 不像本地分类存放

~/Code/predict_protenix 生成decoy的脚本代码

~/Code/Protenix Protenix代码

~/Code/RNA-FM RNA-FM代码

~/Code/RNA_FM_pipeline 用RNA-FM生成残基embedding的脚本代码

~/Code/Split_RNA_dataset 划分数据集的脚本代码

# ~/Code_Flow_matching
模型以及跑的wandb、log所在地

# ~/Data_FM
~/Data_FM/model_cache模型权重

~/Data_FM/RNA_FM_embeddings 2241个残基embedding

# ~/Data_PT_V2
.pt文件  其中验证集和测试集只选200个里面ranking_score最高的

文件结构 ~/Data_PT_V2/{train | val | test | holdout_foldbench_val_homology}/{PDB_id}/{seed_xxx}/.pt

# ~/Data_PT_V2_all_candidates_20261013
.pt文件  验证集和测试集的200个

文件结构 ~/Data_PT_V2_all_candidates_20261013/{val | test}/{PDB_id}/{seed_xxx}/.pt

# ~/Data_PT_V2_RMSD_GT30
额外保存RMSD>30Å的pt 这些pt不进入数据集 但是保存在这里

文件结构 ~/Data_PT_V2_RMSD_GT30/{train | val | test | holdout_foldbench_val_homology}/{PDB_id}/{seed_xxx}/.pt

# ~/Data_V2
第二次划分数据集的产物 包含有置信度的decoy 50*4

文件结构 ~/Data_V2/{train | val | test | holdout_foldbench_val_homology}/{PDB_id}/{seed_xxx}/{predictions}/{.cif 与 置信度文件}

其中为了把foldbench的PDB移动到测试集中，同时保证数据不泄露，把原来验证集中同源的PDB移动到holdout_foldbench_val_homology中了

# ~/Json_data
~/Json_data/Complex_json中有prep_output_xxxx的样式文件夹2227个

~/Json_data/Foldbench_evaluation是评估报告，用到的就是~/Json_data/Foldbench_predictions

~/Json_data/Foldbench_predictions下保存的是5*5的protenix采样，是我们希望用来检验protenix预测RNA构象的能力的

~/Json_data/Simple_json中有xxxx.json的样式文件2241个，对应~/pdb_data中2241个cif 有xxxx-final-updated.json样式文件2227个，表示protenix成功prep的只有2227个

# ~/miniconda3
服务器的conda环境

# ~/output
~/output/ERR包含protenix进行构象生成时错误的PDB_id及其原因

# ~/pdb_data
一共包含2241个.cif(纯RNA)

# ~/pdb_data_mixed
一共包含77个杂牌cif文件。还有一个记录杂牌PDB_id的txt

# ~/protenix_data
~/protenix_data/checkpoint包含protenix的权重protenix_base_default_v1.0.0.pt

~/protenix_data/common包含protenix所需要的common数据

~/protenix_data/rna_msa

~/protenix_data/rna_msa_sequence_cache

~/protenix_data/search_database包含prep进行search的四个数据库

~/protenix_data/search_database_legacy包含之前进行prep进行search的几个数据库(因为复用了1979的数据，所以把和现在不一样的数据库拿过来了)

