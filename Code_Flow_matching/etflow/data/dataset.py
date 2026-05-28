from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset


class EuclideanDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | None = None,
        split: str = "train",
        sample_files: list[str] | None = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.split = split

        if sample_files is not None:
            # 使用指定的文件列表，跳过目录扫描
            self.data_files = [Path(f) for f in sample_files]
        else:
            # 定位到对应的数据集划分目录，例如 /.../RNA/train
            split_dir = self.data_dir / split
            self.data_files = list(split_dir.rglob("*.pt"))

        if len(self.data_files) == 0:
            raise ValueError(
                f"在 {self.data_dir / split if self.data_dir else 'sample_files'} 及其子目录下没有找到任何 .pt 文件！"
            )

        # Sort files for reproducibility
        self.data_files.sort()

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        # Load the data file
        data_path = self.data_files[idx]
        data = torch.load(data_path)

        pos = data['pos']  # [N, 3] 真实的晶体结构坐标 (Ground Truth)
        pos_pred = data['pos_pred']  # [N, 3] Protenix预测的结构坐标 (Condition/Source)
        atomic_numbers = data['atomic_numbers']  # [N] 原子序数
        edge_index = data['edge_index']

        return Data(
            pos=pos,  # 真实坐标 (通常用于计算 Loss)
            pos_pred=pos_pred,  # 预测坐标 (流匹配的起点 / 条件特征)
            z=atomic_numbers,  # PyG 约定的原子序数键名
            edge_index=edge_index,  # 拓扑边
            sequence=data.get('sequence', ''),
            pdb_id=data.get('pdb_id', 'unknown'),
            alignment_ratio=data.get('alignment_ratio', 1.0)  # 方便后续做过滤或权重调整
        )
