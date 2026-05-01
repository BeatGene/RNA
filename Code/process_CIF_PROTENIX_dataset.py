import os
import shutil
import subprocess
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

# ================= 配置区域 =================
# 请在此处修改路径（如果你的路径有变动）
CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
PRED_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"
OUTPUT_BASE = "/remote-home/jinxianwang/tinghaoxia/RNA/Data"

SEEDS = ['42', '43', '44', '45']
CLUSTER_THRESHOLD = 0.8  # 序列相似性阈值 80%，可根据需要调整


# ===========================================

def parse_cif_for_rna(cif_path):
    """
    【增强版】解析 .cif 文件，获取 RNA 链的数量和第一条链的序列
    逻辑向之前的分类脚本靠拢，提高兼容性
    """
    try:
        mmcif_dict = MMCIF2Dict(cif_path)
    except Exception as e:
        print(f"[Warning] 无法解析文件 {cif_path}: {e}")
        return 0, ""

    # 1. 优先从 _entity_poly 入手 (这是之前分类脚本成功的依据)
    # 获取所有 Polymer 的类型和序列
    poly_entity_ids = mmcif_dict.get('_entity_poly.entity_id', [])
    poly_types = mmcif_dict.get('_entity_poly.type', [])
    poly_seqs = mmcif_dict.get('_entity_poly.pdbx_seq_one_letter_code', [])

    # 确保是列表 (有些 CIF 解析出来可能是单字符串)
    if isinstance(poly_entity_ids, str): poly_entity_ids = [poly_entity_ids]
    if isinstance(poly_types, str): poly_types = [poly_types]
    if isinstance(poly_seqs, str): poly_seqs = [poly_seqs]

    # 2. 筛选出 RNA 的 Poly
    rna_poly_indices = []
    for i, ptype in enumerate(poly_types):
        if 'polyribonucleotide' in ptype.lower():
            rna_poly_indices.append(i)

    if not rna_poly_indices:
        # print(f"[Debug] {os.path.basename(cif_path)} 未在 _entity_poly 中找到 polyribonucleotide")
        return 0, ""

    # 3. 计算 RNA 链数 (通过 _struct_asym)
    # 这里我们还是要数一下有多少条链，用于后面的 8:1:1 划分
    asym_entity_ids = mmcif_dict.get('_struct_asym.entity_id', [])
    if isinstance(asym_entity_ids, str): asym_entity_ids = [asym_entity_ids]

    # 找到所有属于 RNA Entity 的 Asym ID
    rna_entity_ids = [poly_entity_ids[i] for i in rna_poly_indices]
    rna_chains_count = sum(1 for eid in asym_entity_ids if eid in rna_entity_ids)

    # 如果通过 asym 没数到链（有些 CIF 这部分可能不规范），
    # 至少我们知道有 RNA，姑且把链数算作 1，保证不丢弃数据
    if rna_chains_count == 0:
        rna_chains_count = 1

    # 4. 获取序列 (拿第一条 RNA Poly 的序列)
    # 直接从刚才筛选出的 rna_poly_indices 里拿第一个
    first_rna_idx = rna_poly_indices[0]
    representative_seq = ""

    if first_rna_idx < len(poly_seqs):
        representative_seq = poly_seqs[first_rna_idx].strip()

    # 有时候 pdbx_seq_one_letter_code 可能是空的，
    # 但我们可以尝试另一个字段: _entity_poly_seq (这是按残基排列的列表)
    if not representative_seq:
        # 尝试从 _entity_poly_seq 拼接 (高级补救措施)
        seq_entity_ids = mmcif_dict.get('_entity_poly_seq.entity_id', [])
        seq_mon_ids = mmcif_dict.get('_entity_poly_seq.mon_id', [])

        target_eid = poly_entity_ids[first_rna_idx]
        # 简单的单字母映射字典 (常用的)
        # 注意：这只是一个简化的补救，不一定覆盖所有修饰碱基
        map_3to1 = {
            'A': 'A', 'ADE': 'A',
            'U': 'U', 'URA': 'U',
            'C': 'C', 'CYT': 'C',
            'G': 'G', 'GUA': 'G',
            'T': 'T', 'THY': 'T'  # 偶尔有 RNA 里写 T 的情况
        }

        rescue_seq = []
        for eid, mon in zip(seq_entity_ids, seq_mon_ids):
            if eid == target_eid:
                rescue_seq.append(map_3to1.get(mon, 'X'))  # X 代表未知

        representative_seq = "".join(rescue_seq)

    # 最终检查：如果还是没序列，只要我们确定它是 RNA (在 Pure_RNA 文件夹里)，
    # 可以考虑返回一个占位符，或者依然返回 0。
    # 这里选择：如果是分类脚本分进来的，即使没序列也给个 'N' 占位，防止数据流失
    if not representative_seq:
        # print(f"[Warning] {os.path.basename(cif_path)} 未找到明确序列，使用占位符")
        representative_seq = "N"

    return rna_chains_count, representative_seq


def main():
    # 1. 扫描并解析所有 CIF 文件
    print("[1/6] 正在扫描并解析 CIF 文件...")
    cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith('.cif')]

    dataset = []  # 元素: (cif_filename, num_chains, seq)
    fasta_content = []

    for cif_file in cif_files:
        cif_path = os.path.join(CIF_DIR, cif_file)
        num_chains, seq = parse_cif_for_rna(cif_path)

        if num_chains > 0 and seq:
            name_id = cif_file.replace('.cif', '')
            dataset.append((cif_file, num_chains, name_id))
            # 构建 FASTA 用于 CD-HIT
            fasta_content.append(f">{name_id}\n{seq}")
        else:
            print(f"[Warning] 跳过 {cif_file} (未找到有效 RNA 链或序列)")

    if not dataset:
        print("[Error] 没有找到有效的 RNA 数据！")
        return

    # 2. 运行 CD-HIT 进行聚类
    print("[2/6] 正在运行序列聚类 (CD-HIT)...")
    fasta_path = "temp_rna_seqs.fasta"
    with open(fasta_path, 'w') as f:
        f.write('\n'.join(fasta_content))

    cdhit_cmd = [
        "cd-hit-est",
        "-i", fasta_path,
        "-o", "cdhit_output",
        "-c", str(CLUSTER_THRESHOLD),
        "-n", "7", "-T", "0", "-M", "0"
    ]

    try:
        subprocess.run(cdhit_cmd, check=True, capture_output=True)
    except FileNotFoundError:
        print("[Error] 未找到 cd-hit-est 命令！请先安装 CD-HIT。")
        return
    except subprocess.CalledProcessError:
        print("[Error] CD-HIT 运行失败，请检查输入序列。")
        return

    # 3. 解析聚类结果 (.clstr 文件)
    print("[3/6] 解析聚类结果...")
    clusters = {}  # cluster_id -> list_of_cif_names
    current_cluster_id = -1

    # 建立 name_id 到完整信息的映射
    info_map = {item[2]: item for item in dataset}  # name_id: (cif_file, num_chains, name_id)

    with open("cdhit_output.clstr", 'r') as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster_id = line.strip().replace(">Cluster ", "")
                clusters[current_cluster_id] = []
            else:
                # 解析类似: 0	100nt, >8d2a... *
                parts = line.split(">")
                if len(parts) > 1:
                    name_id = parts[1].split("...")[0]
                    if name_id in info_map:
                        clusters[current_cluster_id].append(info_map[name_id])

    # 4. 按 8:1:1 划分（基于链数，不拆分 cluster，均衡版）
    print("[4/6] 按 8:1:1 划分数据集...")

    import random
    random.seed(42)

    # 将所有簇打乱（完全随机顺序，避免第一个簇影响过大）
    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids)

    total_chains = sum(item[1] for item in dataset)
    target_ratios = [0.8, 0.1, 0.1]  # 顺序：train, val, test

    # 初始化三个集合
    sets = {
        'train': [],
        'val': [],
        'test': []
    }
    current_chains = {'train': 0, 'val': 0, 'test': 0}

    def assignment_penalty(chain_counts, assign_to, extra_chains, total_after):
        """
        计算如果将 extra_chains 分配给集合 assign_to 后，
        三个集合链数比例的平方误差和（越小越好）。
        """
        # 临时拷贝
        temp = chain_counts.copy()
        temp[assign_to] += extra_chains
        total = sum(temp.values())
        # 防止除以零
        if total == 0:
            return float('inf')

        error = 0
        for i, key in enumerate(['train', 'val', 'test']):
            actual_ratio = temp[key] / total
            error += (actual_ratio - target_ratios[i]) ** 2
        return error

    # 遍历每一个簇，贪心选择误差最小的集合
    for cid in cluster_ids:
        items = clusters[cid]
        cluster_chain_count = sum(item[1] for item in items)

        best_set = None
        best_penalty = float('inf')

        # 尝试分配给 train, val, test
        for candidate in ['train', 'val', 'test']:
            pen = assignment_penalty(
                current_chains,
                candidate,
                cluster_chain_count,
                sum(current_chains.values()) + cluster_chain_count
            )
            if pen < best_penalty:
                best_penalty = pen
                best_set = candidate

        # 将簇分配给最佳集合
        sets[best_set].extend(items)
        current_chains[best_set] += cluster_chain_count

    train_list = sets['train']
    val_list = sets['val']
    test_list = sets['test']

    current_train = current_chains['train']
    current_val = current_chains['val']
    current_test = current_chains['test']
    print(f"    划分完成 (链数): Train={current_train}, Val={current_val}, Test={current_test}")
    # 5. 创建目录结构
    print("[5/6] 创建目录结构...")
    rna_root = os.path.join(OUTPUT_BASE, "RNA")
    dirs_to_create = [
        os.path.join(rna_root, "train"),
        os.path.join(rna_root, "val"),
        os.path.join(rna_root, "test")
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    # 6. 复制文件
    print("[6/6] 复制文件...")

    def copy_files(data_list, target_root):
        for (cif_file, num_chains, name_id) in data_list:
            # 1. 创建真实 CIF 对应的文件夹
            # 例如: .../train/9zcc/
            cif_target_dir = os.path.join(target_root, name_id)
            os.makedirs(cif_target_dir, exist_ok=True)

            # 2. 创建 sample 文件夹
            sample_dir = os.path.join(cif_target_dir, "sample")
            os.makedirs(sample_dir, exist_ok=True)

            # 3. 复制真实的 .cif 文件 (可选，放在 cif_target_dir 下)
            src_cif = os.path.join(CIF_DIR, cif_file)
            shutil.copy2(src_cif, os.path.join(cif_target_dir, cif_file))

            # 4. 复制四个 seed 的预测文件夹
            for seed in SEEDS:
                pred_folder_name = f"pred_output_{name_id}_seed_{seed}"
                src_pred = os.path.join(PRED_DIR, pred_folder_name)
                dst_pred = os.path.join(sample_dir, pred_folder_name)

                if os.path.exists(src_pred):
                    # 使用 dirs_exist_ok=True 防止中断 (Python 3.8+)
                    shutil.copytree(src_pred, dst_pred, dirs_exist_ok=True)
                else:
                    print(f"[Warning] 预测文件夹不存在: {src_pred}")

    copy_files(train_list, dirs_to_create[0])
    copy_files(val_list, dirs_to_create[1])
    copy_files(test_list, dirs_to_create[2])

    # 清理临时文件
    os.remove(fasta_path)
    os.remove("cdhit_output")
    os.remove("cdhit_output.clstr")

    print("全部完成！数据已保存在: ", rna_root)


if __name__ == "__main__":
    main()