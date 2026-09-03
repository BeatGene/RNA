from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset


class EuclideanDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | None = None,
        split: str = "train",
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split

        self.data_files = list((self.data_dir / split).rglob("*.pt"))

        if len(self.data_files) == 0:
            raise ValueError(
                f"在 {self.data_dir / split } 及其子目录下没有找到任何 .pt 文件！"
            )

        # Sort files for reproducibility
        self.data_files.sort()

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        # Load the data file
        data_path = self.data_files[idx]
        data = torch.load(data_path,map_location="cpu",)

        # Modify_3
        pos = data['pos'].float() # [N, 3] 真实的晶体结构坐标 (Ground Truth)
        pos_pred = data['pos_pred'].float()  # [N, 3] Protenix预测的结构坐标 (Condition/Source)
        atomic_numbers = (
            data["atomic_numbers"]
            .long()
            .view(-1)
        )  # [N] 原子序数
        sequence = data['sequence']# RNA的一级序列，比如额AUCGGG
        edge_index = (
            data["edge_index"]
            .long()
            .contiguous()
        )
        # [2,E] 包含:1.核苷酸内部 共价键连接(根据模板)
        #           2.核苷酸之间，添加磷酸二酯键
        #           先不考虑修饰核苷酸
        #           3.空间接触(先只考虑上面两种键)，根据当前坐标计算
        edge_attr = data['edge_attr'].float() # [E,7]
        node_attr = data['node_attr'].float() # [N,M]RNA-FM为每个残基生成特征向量，复制给该残基中每个原子作为背景知识  640维
        # Modify_3
        geometry_bond_index = (
            data["geometry_bond_index"]
            .long()
            .contiguous()
        )

        ideal_bond_length = (
            data["ideal_bond_length"]
            .float()
            .view(-1)
        )

        residue_index = (
            data["residue_index"]
            .long()
            .view(-1)
        )

        atom_name_id = (
            data["atom_name_id"]
            .long()
            .view(-1)
        )

        # 兼容暂时没有 1–3 排除对的旧数据
        clash_exclusion_index = (
            data["clash_exclusion_index"]
            if "clash_exclusion_index" in data
            else geometry_bond_index
        )

        clash_exclusion_index = (
            clash_exclusion_index
            .long()
            .contiguous()
        )

        # ToDo 注意pt文件的生成
        return Data(
            pos=pos,  # 真实坐标 (通常用于计算 Loss) 对应原来数据中的pos
            pos_pred=pos_pred,  # 预测坐标 (流匹配的起点 / 条件特征) 对应原来数据中的采样起点
            atomic_numbers=atomic_numbers,  # 重原子的原子序数 如C是6 N是7 O是8
            sequence=sequence,# RNA的一级序列，对应原来小分子的SMILES字符
            edge_index=edge_index,  # 对应原来数据的edge_index
            edge_attr=edge_attr, # 对应原来数据的edge_attr
            node_attr=node_attr,# 对应原来数据的node_attr
            # Modify_3
            geometry_bond_index=geometry_bond_index,
            ideal_bond_length=ideal_bond_length,
            residue_index=residue_index,
            atom_name_id=atom_name_id,
            clash_exclusion_index=clash_exclusion_index,
        )
