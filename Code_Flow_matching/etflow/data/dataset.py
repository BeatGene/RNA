from pathlib import Path

import torch
# Modify_4
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset

from etflow.data.constants import (
    NUM_ATOM_NAME_TYPES,
    NUM_RNA_RESIDUE_TYPES,
    RNAFM_EMBEDDING_DIM,
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
        # Modify_4

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
        #Modify_4
        rnafm_embedding = (
            data["rnafm_embedding"]
            .float()
            .contiguous()
        )

        residue_type_id = (
            data["residue_type_id"]
            .long()
            .view(-1)
        )

        if rnafm_embedding.dim() != 2:
            raise ValueError(
                "rnafm_embedding must have shape [num_residues, 640], "
                f"but got {tuple(rnafm_embedding.shape)}"
            )

        if rnafm_embedding.size(1) != RNAFM_EMBEDDING_DIM:
            raise ValueError(
                f"rnafm_embedding has dim {rnafm_embedding.size(1)}, "
                f"but expected {RNAFM_EMBEDDING_DIM}"
            )

        num_residues = rnafm_embedding.size(0)

        if residue_type_id.numel() != num_residues:
            raise ValueError(
                "residue_type_id and rnafm_embedding disagree on "
                f"num_residues: {residue_type_id.numel()} versus "
                f"{num_residues}"
            )

        if residue_index.numel() != pos.size(0):
            raise ValueError(
                "residue_index must contain one value per atom"
            )

        if (
                residue_index.numel() > 0
                and (
                residue_index.min() < 0
                or residue_index.max() >= num_residues
        )
        ):
            raise ValueError(
                "residue_index contains an invalid local residue id"
            )

        if (
                atom_name_id.min() < 0
                or atom_name_id.max() >= NUM_ATOM_NAME_TYPES
        ):
            raise ValueError(
                "atom_name_id contains an invalid atom-name id"
            )

        if (
                residue_type_id.min() < 0
                or residue_type_id.max() >= NUM_RNA_RESIDUE_TYPES
        ):
            raise ValueError(
                "residue_type_id contains an invalid RNA residue id"
            )

        rnafm_atom = rnafm_embedding[residue_index]

        atom_role_one_hot = F.one_hot(
            atom_name_id,
            num_classes=NUM_ATOM_NAME_TYPES,
        ).float()

        residue_type_atom = residue_type_id[residue_index]

        residue_type_one_hot = F.one_hot(
            residue_type_atom,
            num_classes=NUM_RNA_RESIDUE_TYPES,
        ).float()

        node_attr = torch.cat(
            [
                rnafm_atom,
                atom_role_one_hot,
                residue_type_one_hot,
            ],
            dim=-1,
        ).contiguous()

        atom_plddt = (
            data["atom_plddt"]
            .float()
            .view(-1)
        )

        atom_to_token_idx = (
            data["atom_to_token_idx"]
            .long()
            .view(-1)
        )

        token_pair_pae = (
            data["token_pair_pae"]
            .float()
            .contiguous()
        )

        token_pair_pde = (
            data["token_pair_pde"]
            .float()
            .contiguous()
        )

        contact_probs = (
            data["contact_probs"]
            .float()
            .contiguous()
        )

        if atom_plddt.numel() != pos.size(0):
            raise ValueError(
                "atom_plddt must contain one value per atom"
            )

        if atom_to_token_idx.numel() != pos.size(0):
            raise ValueError(
                "atom_to_token_idx must contain one value per atom"
            )

        if token_pair_pae.dim() != 2:
            raise ValueError(
                "token_pair_pae must have shape [num_tokens, num_tokens]"
            )

        num_tokens = token_pair_pae.size(0)

        expected_pair_shape = (num_tokens, num_tokens)

        for feature_name, feature_value in {
            "token_pair_pae": token_pair_pae,
            "token_pair_pde": token_pair_pde,
            "contact_probs": contact_probs,
        }.items():
            if tuple(feature_value.shape) != expected_pair_shape:
                raise ValueError(
                    f"{feature_name} has shape "
                    f"{tuple(feature_value.shape)}, expected "
                    f"{expected_pair_shape}"
                )

        if (
                atom_to_token_idx.numel() > 0
                and (
                atom_to_token_idx.min() < 0
                or atom_to_token_idx.max() >= num_tokens
        )
        ):
            raise ValueError(
                "atom_to_token_idx contains an invalid local token id"
            )

        if not torch.isfinite(atom_plddt).all():
            raise ValueError("atom_plddt contains NaN or Inf")

        if not torch.isfinite(token_pair_pae).all():
            raise ValueError("token_pair_pae contains NaN or Inf")

        if not torch.isfinite(token_pair_pde).all():
            raise ValueError("token_pair_pde contains NaN or Inf")

        if not torch.isfinite(contact_probs).all():
            raise ValueError("contact_probs contains NaN or Inf")

        if atom_plddt.min() < 0 or atom_plddt.max() > 1:
            raise ValueError(
                "atom_plddt must be normalized to [0, 1]"
            )

        if contact_probs.min() < 0 or contact_probs.max() > 1:
            raise ValueError(
                "contact_probs must be in [0, 1]"
            )

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

        # ToDo 注意pt文件的生成
        return RNAData(
            pos=pos,
            pos_pred=pos_pred,
            atomic_numbers=atomic_numbers,
            sequence=sequence,
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
        )
