import os
import shutil
import random
import subprocess
import warnings
from tqdm import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

# 忽略 Biopython 解析时的警告信息，保持控制台整洁
warnings.simplefilter('ignore', PDBConstructionException)
warnings.filterwarnings('ignore', category=UserWarning)


def check_cdhit_installed():
    """检查系统是否安装了 cd-hit-est"""
    if shutil.which("cd-hit-est") is None:
        raise EnvironmentError(
            "未找到 'cd-hit-est' 命令！\n"
            "为了防止数据泄露，必须先进行聚类。请在终端执行以下命令安装：\n"
            "conda install -c bioconda cd-hit"
        )


def extract_sequences_to_fasta(src_dir, fasta_path):
    """鲁棒地从 CIF 文件中提取序列，包含双重提取机制"""
    print("步骤 1: 正在从 CIF 文件中提取 RNA 序列 (启动双重提取机制)...")
    cif_files = [f for f in os.listdir(src_dir) if f.endswith('.cif')]

    parser = MMCIFParser(QUIET=True)

    success_count = 0
    fail_count = 0
    seq_dict = {}  # 核心改动：用字典记录所有成功提取的序列

    with open(fasta_path, 'w') as f_out:
        for filename in tqdm(cif_files, desc="提取序列"):
            filepath = os.path.join(src_dir, filename)
            pdb_id = filename.split('.')[0]
            full_seq = ""

            # 【提取方法 1】：字典极速提取（读取元数据）
            try:
                cif_dict = MMCIF2Dict(filepath)
                seqs = cif_dict.get('_entity_poly.pdbx_seq_one_letter_code', [])
                if isinstance(seqs, str):
                    seqs = [seqs]
                full_seq = "".join([s.replace('\n', '').replace(' ', '') for s in seqs])
            except Exception:
                pass

            # 【提取方法 2】：直接解析 3D 坐标提取残基名
            if not full_seq or len(full_seq) == 0:
                try:
                    structure = parser.get_structure(pdb_id, filepath)
                    seq_list = []
                    for model in structure:
                        for chain in model:
                            for res in chain:
                                if res.id[0] == " ":
                                    resname = res.get_resname().strip()
                                    if resname:
                                        seq_list.append(resname[0])
                        break
                    full_seq = "".join(seq_list)
                except Exception:
                    pass

            # 写入 FASTA 并且存入字典
            if full_seq and len(full_seq) > 0:
                f_out.write(f">{pdb_id}\n{full_seq}\n")
                seq_dict[pdb_id] = full_seq
                success_count += 1
            else:
                fail_count += 1

    print(f"\n✅ 序列提取完成！成功提取: {success_count} 个，彻底失败: {fail_count} 个。")
    return seq_dict


def run_cdhit_clustering(fasta_path, out_prefix, identity_threshold=0.8):
    """运行 cd-hit-est 进行聚类"""
    print(f"\n步骤 2: 正在运行 CD-HIT-EST (相似度阈值: {identity_threshold * 100}%)...")
    cmd = [
        "cd-hit-est",
        "-i", fasta_path,
        "-o", out_prefix,
        "-c", str(identity_threshold),
        "-n", "5",
        "-M", "0",
        "-d", "0",
        "-l", "5"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("CD-HIT 运行出错:\n", result.stderr)
        raise RuntimeError("CD-HIT 聚类失败。")
    print("CD-HIT 聚类完成！")


def parse_cdhit_clusters(clstr_path):
    """解析 .clstr 文件，返回簇的列表"""
    print("\n步骤 3: 解析聚类结果...")
    clusters = []
    current_cluster = []
    clustered_ids = set()

    with open(clstr_path, 'r') as f:
        for line in f:
            if line.startswith(">Cluster"):
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
            else:
                parts = line.split(">", 1)
                if len(parts) > 1:
                    pdb_id = parts[1].split()[0].replace('...', '').strip()
                    current_cluster.append(pdb_id)
                    clustered_ids.add(pdb_id)

    if current_cluster:
        clusters.append(current_cluster)

    print(f"CD-HIT 共生成 {len(clusters)} 个独立的 RNA 簇，涵盖 {len(clustered_ids)} 个结构。")
    return clusters, clustered_ids


def split_dataset_by_cluster(dst_dir, clusters, all_original_ids, clustered_ids, train_ratio=0.8, val_ratio=0.1,
                             test_ratio=0.1, seed=42):
    """基于聚类结果进行划分，并处理游离/失败文件"""
    print("\n步骤 4: 正在按簇分配数据集，并创建对应文件夹...")
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5

    random.seed(seed)
    random.shuffle(clusters)

    total_clustered_files = sum(len(c) for c in clusters)
    target_counts = {
        'train': int(total_clustered_files * train_ratio),
        'val': int(total_clustered_files * val_ratio),
        'test': total_clustered_files - int(total_clustered_files * train_ratio) - int(
            total_clustered_files * val_ratio)
    }

    clusters.sort(key=len, reverse=True)

    current_counts = {'train': 0, 'val': 0, 'test': 0}
    split_dict = {'train': [], 'val': [], 'test': []}

    # 贪心分配
    for cluster in clusters:
        best_split = None
        max_deficit = -float('inf')
        for split in ['train', 'val', 'test']:
            deficit = target_counts[split] - current_counts[split]
            if deficit > max_deficit:
                max_deficit = deficit
                best_split = split

        split_dict[best_split].extend(cluster)
        current_counts[best_split] += len(cluster)

    missing_ids = set(all_original_ids) - clustered_ids

    splits = ['train', 'val', 'test']
    dst_paths = {split: os.path.join(dst_dir, split) for split in splits}
    failed_path = os.path.join(dst_dir, "failed_or_skipped")

    for path in dst_paths.values():
        os.makedirs(path, exist_ok=True)

    print("\n--- 最终有效数据划分统计 ---")
    for split in splits:
        actual_ratio = current_counts[split] / total_clustered_files if total_clustered_files > 0 else 0
        print(f"{split.capitalize()} 集: 分配了 {current_counts[split]} 个文件 (占比 {actual_ratio * 100:.1f}%)")
    print("--------------------\n")

    # 在执行创建文件夹前，先清理一下可能存在的旧的 failed 文件夹，防止干扰视线
    if os.path.exists(failed_path):
        try:
            shutil.rmtree(failed_path)
        except:
            pass

    for split in splits:
        target_dir = dst_paths[split]
        for pdb_id in tqdm(split_dict[split], desc=f"在 {split} 集中创建文件夹"):
            folder_path = os.path.join(target_dir, pdb_id)
            os.makedirs(folder_path, exist_ok=True)

    if missing_ids:
        os.makedirs(failed_path, exist_ok=True)
        print(f"\n⚠️ 发现 {len(missing_ids)} 个因严重异常未能提取序列的文件。")
        for pdb_id in tqdm(missing_ids, desc="记录未划分结构"):
            folder_path = os.path.join(failed_path, pdb_id)
            os.makedirs(folder_path, exist_ok=True)
        print(f"这部分文件已放入 {failed_path} 中。")

    print(
        f"\n✅ 全部处理完毕！总文件数核对: 成功划分 {total_clustered_files} + 异常丢弃 {len(missing_ids)} = {total_clustered_files + len(missing_ids)}")


if __name__ == "__main__":
    SRC_DIRECTORY = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
    DST_DIRECTORY = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"

    TEMP_DIR = os.path.join(DST_DIRECTORY, "clustering_temp")
    os.makedirs(TEMP_DIR, exist_ok=True)

    FASTA_FILE = os.path.join(TEMP_DIR, "all_rna.fasta")
    CLSTR_PREFIX = os.path.join(TEMP_DIR, "rna_clustered")
    CLSTR_FILE = f"{CLSTR_PREFIX}.clstr"

    check_cdhit_installed()

    # 1. 提取所有序列到字典
    seq_dict = extract_sequences_to_fasta(SRC_DIRECTORY, FASTA_FILE)

    # 2. CD-HIT 聚类 (它会自动丢弃过短的序列)
    run_cdhit_clustering(FASTA_FILE, CLSTR_PREFIX, identity_threshold=0.8)

    # 3. 解析 CD-HIT 结果
    clusters, clustered_ids = parse_cdhit_clusters(CLSTR_FILE)

    # ================= 核心修复逻辑 =================
    missing_ids = set(seq_dict.keys()) - clustered_ids
    if missing_ids:
        print(f"\n⚠️ 检测到 {len(missing_ids)} 个由于长度过短被 CD-HIT 忽略的 RNA。启动原生序列精确匹配进行抢救...")
        short_clusters_map = {}
        for pid in missing_ids:
            seq = seq_dict[pid].upper()  # 转大写匹配
            if seq not in short_clusters_map:
                short_clusters_map[seq] = []
            short_clusters_map[seq].append(pid)

        short_clusters = list(short_clusters_map.values())
        print(f"✅ 抢救成功！已将这 {len(missing_ids)} 个短序列归纳为 {len(short_clusters)} 个新簇。")

        # 并入总簇列表
        clusters.extend(short_clusters)
        clustered_ids.update(missing_ids)
    # ================================================

    # 4. 最终划分 (现在的 clusters 包含了长短序列所有的簇)
    split_dataset_by_cluster(
        dst_dir=DST_DIRECTORY,
        clusters=clusters,
        all_original_ids=list(seq_dict.keys()),
        clustered_ids=clustered_ids,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)