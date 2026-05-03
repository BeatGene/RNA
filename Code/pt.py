import os
import torch
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import MMCIFParser

# 忽略生物库的警告
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionException

warnings.simplefilter('ignore', PDBConstructionException)

# 定义元素的原子序数映射表（排除氢）
ELEMENT_TO_Z = {'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16, 'MG': 12, 'K': 19}


def native_radius_graph(x, r):
    """原生 PyTorch 实现的 radius_graph"""
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(float('inf'))
    mask = dist < r
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)


def extract_atoms_from_cif(cif_path, structure_id="id"):
    """
    双重索引提取：同时返回严格索引(exact)和序列软索引(seq)
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_id, cif_path)
    except Exception as e:
        return None, None, None

    exact_dict = {}
    seq_dict = {}
    sequence_list = []

    model = structure[0]
    global_res_idx = 0  # 追踪绝对的序列顺序

    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue

            res_name = res.get_resname().strip()
            res_id = res.id[1]
            sequence_list.append(res_name[0] if res_name else 'N')

            for atom in res:
                element = atom.element.upper()
                if element == 'H':
                    continue

                atom_name = atom.get_name().strip()

                # 严格 ID (依赖 PDB 原生链和残基号)
                exact_id = f"{chain.id}_{res_id}_{atom_name}"
                # 软对齐 ID (仅依赖序列出现的先后顺序，破解 Protenix 编号重置问题)
                seq_id = f"{global_res_idx}_{atom_name}"

                val = {
                    'pos': atom.get_coord(),
                    'element': element,
                    'res_name': res_name
                }

                exact_dict[exact_id] = val
                seq_dict[seq_id] = val

            global_res_idx += 1

    seq_str = "".join(sequence_list)
    return exact_dict, seq_dict, seq_str


def process_single_pdb(pdb_id, gt_cif_path, pred_base_dir, save_dir):
    gt_exact, gt_seq, sequence = extract_atoms_from_cif(gt_cif_path, pdb_id)
    if gt_exact is None:
        return 0, "无法解析真实的 CIF 文件(可能文件损坏)"

    pt_saved_count = 0
    seeds = [42, 43, 44, 45]

    found_any_pred_folder = False
    found_any_pred_cif = False

    for seed in seeds:
        pred_folder = os.path.join(pred_base_dir, f"pred_output_{pdb_id}_seed_{seed}", pdb_id, f"seed_{seed}",
                                   "predictions")
        if not os.path.exists(pred_folder):
            continue
        found_any_pred_folder = True

        pred_cifs = [f for f in os.listdir(pred_folder) if f.endswith('.cif')]
        if not pred_cifs:
            continue
        found_any_pred_cif = True

        for pred_cif in pred_cifs:
            pred_cif_path = os.path.join(pred_folder, pred_cif)
            pred_exact, pred_seq, _ = extract_atoms_from_cif(pred_cif_path, f"{pdb_id}_pred")
            if not pred_exact:
                continue

            aligned_gt_pos = []
            aligned_pred_pos = []
            atomic_numbers = []

            # 【策略 A】: 严格对齐尝试
            for atom_id, gt_info in gt_exact.items():
                if atom_id in pred_exact:
                    element = gt_info['element']
                    z = ELEMENT_TO_Z.get(element, 0)
                    if z == 0: continue
                    aligned_gt_pos.append(gt_info['pos'])
                    aligned_pred_pos.append(pred_exact[atom_id]['pos'])
                    atomic_numbers.append(z)

            # 【策略 B】: 降维打击！如果严格对齐失败，启动序列软对齐
            if len(aligned_gt_pos) < 10:
                aligned_gt_pos = []
                aligned_pred_pos = []
                atomic_numbers = []
                for atom_id, gt_info in gt_seq.items():
                    if atom_id in pred_seq:
                        element = gt_info['element']
                        z = ELEMENT_TO_Z.get(element, 0)
                        if z == 0: continue
                        aligned_gt_pos.append(gt_info['pos'])
                        aligned_pred_pos.append(pred_seq[atom_id]['pos'])
                        atomic_numbers.append(z)

            # 最终检查
            if len(aligned_gt_pos) < 10:
                continue

            alignment_ratio = len(aligned_gt_pos) / len(gt_exact) if len(gt_exact) > 0 else 0.0

            pos_tensor = torch.tensor(aligned_gt_pos, dtype=torch.float)
            pos_pred_tensor = torch.tensor(aligned_pred_pos, dtype=torch.float)
            z_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
            edge_index = native_radius_graph(pos_pred_tensor, r=4.5)

            data_dict = {
                'pos': pos_tensor,
                'pos_pred': pos_pred_tensor,
                'atomic_numbers': z_tensor,
                'sequence': sequence,
                'edge_index': edge_index,
                'pdb_id': pdb_id,
                'seed': seed,
                'pred_filename': pred_cif,
                'alignment_ratio': alignment_ratio
            }

            save_name = f"{pdb_id}_s{seed}_{pred_cif.replace('.cif', '.pt')}"
            torch.save(data_dict, os.path.join(save_dir, save_name))
            pt_saved_count += 1

    if pt_saved_count == 0:
        if not found_any_pred_folder:
            # 抛出具体的路径，方便你去查是不是拼错了或者少了个 seed
            sample_path = os.path.join(pred_base_dir, f"pred_output_{pdb_id}_seed_42", pdb_id, "seed_42", "predictions")
            return 0, f"找不到预测文件夹 (预期路径类似: {sample_path})"
        elif not found_any_pred_cif:
            return 0, "预测文件夹存在，但里面没有找到任何 .cif 结果文件"
        else:
            return 0, "预测文件存在，但软硬对齐均失败(原子数量差异过大或命名严重畸变)"

    return pt_saved_count, "成功"


def main(prefix_filter=None):
    SPLITS = ['train', 'val', 'test']
    ORIGINAL_CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
    PRED_BASE_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"
    SAVE_BASE_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"

    total_generated = 0
    failed_pdbs = []

    for split in SPLITS:
        print(f"\n🚀 开始处理 {split} 集数据...")
        split_dir = os.path.join(SAVE_BASE_DIR, split)

        if not os.path.exists(split_dir):
            continue

        pdb_folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]

        if prefix_filter:
            pdb_folders = [f for f in pdb_folders if f.startswith(prefix_filter)]
            print(f"🔍 仅处理以 '{prefix_filter}' 开头的 PDB，本集共 {len(pdb_folders)} 个。")

        for pdb_id in tqdm(pdb_folders, desc=f"Processing {split}"):
            gt_cif_path = os.path.join(ORIGINAL_CIF_DIR, f"{pdb_id}.cif")

            if not os.path.exists(gt_cif_path):
                failed_pdbs.append(f"{pdb_id} (Reason: 找不到真实 cif 文件)")
                continue

            save_dir = os.path.join(split_dir, pdb_id)
            count, reason = process_single_pdb(pdb_id, gt_cif_path, PRED_BASE_DIR, save_dir)
            total_generated += count

            if count == 0:
                failed_pdbs.append(f"{pdb_id} (Reason: {reason})")

    print(f"\n✅ 所有数据处理完毕！共生成 {total_generated} 个 .pt 训练样本。")

    if failed_pdbs:
        log_name = f"failed_generation_{prefix_filter if prefix_filter else 'all'}.txt"
        failed_log_path = os.path.join(SAVE_BASE_DIR, log_name)
        with open(failed_log_path, "w") as f:
            for item in failed_pdbs:
                f.write(f"{item}\n")
        print(f"⚠️ 有 {len(failed_pdbs)} 个 PDB 文件生成失败/被跳过，精确报错已记录在: {failed_log_path}")


if __name__ == "__main__":
    main(prefix_filter="1")