from pathlib import Path

import torch
# Modify_4
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset

from etflow.data.constants import (
    NUM_ATOM_NAME_TYPES,
    NUM_RNA_RESIDUE_TYPES,
)


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
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.include_metadata = include_metadata

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

        rnafm_embedding = data["rnafm_embedding"].float().contiguous()
        residue_type_id = data["residue_type_id"].long().view(-1)
        rnafm_atom = rnafm_embedding[residue_index]

        atom_role_one_hot = F.one_hot(atom_name_id,num_classes=NUM_ATOM_NAME_TYPES,).float()
        residue_type_atom = residue_type_id[residue_index]
        residue_type_one_hot = F.one_hot(residue_type_atom,num_classes=NUM_RNA_RESIDUE_TYPES,).float()

        node_attr = torch.cat([rnafm_atom,atom_role_one_hot,residue_type_one_hot,],dim=-1,).contiguous()

        atom_plddt = data["atom_plddt"].float().view(-1)
        atom_to_token_idx = data["atom_to_token_idx"].long().view(-1)
        token_pair_pae = data["token_pair_pae"].float().contiguous()
        token_pair_pde = data["token_pair_pde"].float().contiguous()
        contact_probs = data["contact_probs"].float().contiguous()

        num_tokens = token_pair_pae.size(0)

        # Protenix当前PAE/PDE分箱范围是0–32 Å。
        token_pair_confidence = torch.stack(
            [
                token_pair_pae.clamp(0.0, 32.0) / 32.0,
                token_pair_pae.transpose(0, 1).clamp(0.0, 32.0) / 32.0,
                token_pair_pde.clamp(0.0, 32.0) / 32.0,
                contact_probs.clamp(0.0, 1.0),
            ],
            dim=-1,
        ).reshape(-1, 4).contiguous()

        # Modify_5
        token_mobility_attr = torch.stack(
            [
                token_pair_pae.mean(dim=1).clamp(0.0, 32.0) / 32.0,
                token_pair_pae.mean(dim=0).clamp(0.0, 32.0) / 32.0,
                token_pair_pde.mean(dim=1).clamp(0.0, 32.0) / 32.0,
                contact_probs.mean(dim=1).clamp(0.0, 1.0),
            ],
            dim=-1,
        )

        atom_mobility_attr = token_mobility_attr[atom_to_token_idx].contiguous()

        result = RNAData(
            pos=pos,
            pos_pred=pos_pred,
            target_mask=target_mask,
            atomic_numbers=atomic_numbers,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            geometry_bond_index=geometry_bond_index,
            ideal_bond_length=ideal_bond_length,
            residue_index=residue_index,
            atom_name_id=atom_name_id,
            clash_exclusion_index=clash_exclusion_index,
            # Modify_4
            atom_plddt=atom_plddt,
            atom_to_token_idx=atom_to_token_idx,
            token_pair_confidence=token_pair_confidence,
            num_tokens=torch.tensor(
                [num_tokens],
                dtype=torch.long,
            ),
            # Modify_5
            atom_mobility_attr=atom_mobility_attr,
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
