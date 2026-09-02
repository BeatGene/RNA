import os
import re
import torch
from pathlib import Path
from tqdm import tqdm


def find_rigid_alignment(A, B):
    """
    Kabsch algorithm: align point cloud A to reference B.
    Returns rotation R and translation t such that (R @ A.T).T + t ≈ B.
    """
    a_mean = A.mean(dim=0)
    b_mean = B.mean(dim=0)
    A_c = A - a_mean
    B_c = B - b_mean
    H = A_c.T @ B_c
    U, S, V = torch.svd(H)
    R = V @ U.T
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = b_mean - R @ a_mean
    return R, t


def load_low_res_ids(txt_path):
    """从 low_resolution_files.txt 读取低分辨率 PDB ID 集合（小写）。"""
    ids = set()
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 格式: 1fcw.cif	1FCW	17.0 Å
            parts = line.split()
            if parts:
                filename = parts[0]
                pdb_id = filename.replace('.cif', '').lower()
                ids.add(pdb_id)
    return ids


def extract_pdb_id(pt_filename):
    """从 .pt 文件名提取 PDB ID。例如 9cso_s45_9cso_sample_9.pt → 9cso"""
    # PDB ID 固定 4 字符，取文件名前 4 位
    return pt_filename[:4]


def align_file(pt_path):
    """对单个 .pt 文件做刚性对齐，pos_pred 对齐到 pos。返回是否成功。"""
    try:
        data = torch.load(pt_path, weights_only=True)
    except Exception:
        return False

    pos = data['pos']
    pos_pred = data['pos_pred']

    if pos.shape != pos_pred.shape or pos.shape[0] < 3:
        return False

    R, t = find_rigid_alignment(pos_pred, pos)
    pos_pred_aligned = (R @ pos_pred.T).T + t

    data['pos_pred'] = pos_pred_aligned
    torch.save(data, pt_path)
    return True


def main():
    DATA_ROOT = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"
    LOW_RES_TXT = "/remote-home/jinxianwang/tinghaoxia/RNA/Code/Ana_pure_RNA/low_resolution_files.txt"
    SPLITS = ['train', 'val', 'test']

    # 加载低分辨率 ID 列表
    low_res_ids = load_low_res_ids(LOW_RES_TXT)
    print(f"低分辨率 PDB ID: {len(low_res_ids)} 个")
    for rid in sorted(low_res_ids):
        print(f"  {rid}")

    total = 0
    aligned = 0
    deleted = 0
    failed = []

    for split in SPLITS:
        split_dir = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_dir):
            continue

        pt_files = list(Path(split_dir).rglob("*.pt"))
        print(f"\n{split}: 找到 {len(pt_files)} 个 .pt 文件")

        for pt_path in tqdm(pt_files, desc=f"处理 {split}"):
            total += 1
            path_str = str(pt_path)
            filename = os.path.basename(path_str)
            pdb_id = extract_pdb_id(filename)

            # 低分辨率 → 直接删除
            if pdb_id in low_res_ids:
                os.remove(path_str)
                deleted += 1
                continue

            # 正常文件 → 对齐
            if align_file(path_str):
                aligned += 1
            else:
                failed.append(path_str)

    print(f"\n{'='*50}")
    print(f"总计: {total} 个文件")
    print(f"  对齐成功: {aligned}")
    print(f"  低分辨率已删除: {deleted}")
    print(f"  失败: {len(failed)}")
    if failed:
        for f in failed[:20]:
            print(f"  {f}")
        if len(failed) > 20:
            print(f"  ... 共 {len(failed)} 个")


if __name__ == "__main__":
    main()
