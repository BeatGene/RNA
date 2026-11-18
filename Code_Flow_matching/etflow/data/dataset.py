from __future__ import annotations
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset


# Modify_4
class RNAData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        # 这两个编号都是每个RNA样本内部的局部编号，
        # 不能按照原子数自动增加。
        if key in {
            "residue_index",
            "atom_to_token_idx",
        }:
            return 0

        return super().__inc__(
            key,
            value,
            *args,
            **kwargs,
        )


class EuclideanDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | None = None,
        split: str = "train",
        include_metadata: bool = False,
        confidence_mode: str = "auto",
        validate_inputs: bool = True,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.include_metadata = include_metadata
        self.confidence_mode = confidence_mode
        self.validate_inputs = validate_inputs

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

        if self.validate_inputs:
            from .validation import validate_record
            validate_record(data)

        # Modify_3
        pos = data['pos'].float() # [N, 3] 真实的晶体结构坐标 (Ground Truth)
        pos_pred = data['pos_pred'].float()  # [N, 3] Protenix预测的结构坐标 (Condition/Source)
        target_mask = data["target_mask"].bool().view(-1)

        atomic_numbers = data["atomic_numbers"].long().view(-1)  # [N] 原子序数
        edge_index = data["edge_index"].long().contiguous()
        # [2,E] 包含:1.核苷酸内部 共价键连接(根据模板)
        #           2.核苷酸之间，添加磷酸二酯键
        #           先不考虑修饰核苷酸
        #           3.空间接触(先只考虑上面两种键)，根据当前坐标计算

        edge_attr = data['edge_attr'].float() # [E,7]
        geometry_bond_index = data["geometry_bond_index"].long().contiguous()
        ideal_bond_length = data["ideal_bond_length"].float().view(-1)

        residue_index = data["residue_index"].long().view(-1)
        atom_name_id = data["atom_name_id"].long().view(-1)
        clash_exclusion_index = data["clash_exclusion_index"] if "clash_exclusion_index" in data else geometry_bond_index

        clash_exclusion_index = clash_exclusion_index.long().contiguous()

        from .features import node_features
        node_attr = node_features(data, residue_index, atom_name_id)

        from .confidence import confidence_features
        confidence = confidence_features(data, residue_index, self.confidence_mode)

        result = RNAData(
            pos=pos,            #[N,3]
            pos_pred=pos_pred,  #[N,3]
            target_mask=target_mask, #[N]
            atomic_numbers=atomic_numbers, #[N]
            edge_index=edge_index,    #[2,E]
            edge_attr=edge_attr,      #[E,7]
            node_attr=node_attr,      #[N,640+29+5]
            geometry_bond_index=geometry_bond_index, #[2,G]
            ideal_bond_length=ideal_bond_length,     #[G]
            residue_index=residue_index,             #[N]
            atom_name_id=atom_name_id,               #[N]
            clash_exclusion_index=clash_exclusion_index,# [2,G] or [2,C]
            **confidence,
        )

        # Training does not need Python/string metadata, but evaluation needs a
        # stable mapping from every prediction back to its PDB/chain/candidate.
        # PyG batches string attributes as lists and leaves them on the CPU.
        if self.include_metadata:
            result.sample_id = str(data.get("sample_id", data_path.stem))
            result.native_structure_id = str(
                data.get("native_structure_id", data_path.parents[1].name)
            )
            result.native_chain_id = str(data.get("native_chain_id", ""))
            result.predicted_chain_id = str(data.get("predicted_chain_id", ""))
            result.sample_path = str(data_path)
            result.protenix_seed = torch.tensor(
                [int(data.get("protenix_seed", -1))], dtype=torch.long
            )
            result.protenix_sample = torch.tensor(
                [int(data.get("protenix_sample", -1))], dtype=torch.long
            )
            result.sequence_length = torch.tensor(
                [len(str(data.get("sequence", "")))], dtype=torch.long
            )
            result.stored_input_rmsd = torch.tensor(
                [float(data.get("pre_refinement_aligned_rmsd", float("nan")))],
                dtype=torch.float32,
            )

        return result
