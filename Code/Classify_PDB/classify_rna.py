import os
import shutil
from tqdm import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import warnings

# 忽略 Biopython 的烦人警告
warnings.filterwarnings('ignore', category=UserWarning)


def classify_cif_files(data_dir="./pdb_data"):
    """
    极速对 CIF 文件进行分类并移动到对应的子文件夹
    """
    # 1. 定义目标文件夹路径
    dirs = {
        "pure_rna": os.path.join(data_dir, "01_Pure_RNA"),
        "rna_protein": os.path.join(data_dir, "02_RNA_Protein_Complex"),
        "ribosome": os.path.join(data_dir, "03_Ribosome_Apo"),
        "ribosome_bound": os.path.join(data_dir, "04_Ribosome_Bound_RNA"),
        "others": os.path.join(data_dir, "05_Others_or_Failed")
    }

    # 创建这些文件夹
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 获取所有 cif 文件（只扫描当前目录下的文件，不包含子文件夹）
    cif_files = [f for f in os.listdir(data_dir) if f.endswith('.cif')]

    print(f"总计检测到 {len(cif_files)} 个 .cif 文件，开始极速分类...")

    # 统计计数器
    stats = {k: 0 for k in dirs.keys()}

    for filename in tqdm(cif_files, desc="分类进度"):
        filepath = os.path.join(data_dir, filename)

        try:
            # 高效读取机制：只解析 CIF 字典，不构建 3D 结构树，速度快百倍
            mmcif_dict = MMCIF2Dict(filepath)

            # --- 提取并格式化所需字段 ---
            # 1. 标题 (Title)
            title = mmcif_dict.get('_struct.title', [''])[0].lower()

            # 2. 分子类型 (Polymer Types)
            poly_types = mmcif_dict.get('_entity_poly.type', [])
            if isinstance(poly_types, str): poly_types = [poly_types]
            poly_types_str = " ".join([str(t).lower() for t in poly_types])

            # 3. 分子描述 (Entity Descriptions - 用于识别 tRNA/mRNA)
            entity_desc = mmcif_dict.get('_entity.pdbx_description', [])
            if isinstance(entity_desc, str): entity_desc = [entity_desc]
            entity_desc_str = " ".join([str(d).lower() for d in entity_desc])

            # --- 逻辑判断标志位 ---
            has_rna = 'polyribonucleotide' in poly_types_str
            has_protein = 'polypeptide' in poly_types_str
            has_dna = 'polydeoxyribonucleotide' in poly_types_str

            # 核糖体关键词匹配
            ribosome_keywords = ['ribosome', 'ribosomal', '50s', '30s', '70s', '80s', '40s', '60s']
            is_ribosome = any(kw in title for kw in ribosome_keywords) or any(
                kw in entity_desc_str for kw in ribosome_keywords)

            # 结合态RNA匹配 (tRNA, mRNA, sgRNA 等)
            bound_rna_keywords = ['trna', 'mrna', 'transfer rna', 'messenger rna', 'aptamer']
            has_bound_rna = any(kw in entity_desc_str for kw in bound_rna_keywords) or any(
                kw in title for kw in bound_rna_keywords)

            # --- 分类路由逻辑 ---
            target_key = "others"

            if is_ribosome:
                if has_bound_rna:
                    target_key = "ribosome_bound"  # 包含tRNA/mRNA的核糖体
                else:
                    target_key = "ribosome"  # 纯核糖体 (rRNA + rProteins)
            elif has_protein and has_rna:
                target_key = "rna_protein"  # 普通的 RNA-蛋白质复合体
            elif has_rna and not has_protein and not has_dna:
                target_key = "pure_rna"  # 纯 RNA (且不含DNA)
            else:
                target_key = "others"  # 包含 DNA-RNA 杂交链，或解析异常的文件

            # 执行物理移动文件
            shutil.move(filepath, os.path.join(dirs[target_key], filename))
            stats[target_key] += 1

        except Exception as e:
            # 遇到极个别损坏的 CIF 文件，直接扔进 others
            shutil.move(filepath, os.path.join(dirs["others"], filename))
            stats["others"] += 1

    # 打印最终统计结果
    print("\n✅ 分类完成！统计结果如下：")
    print(f"📁 01_纯 RNA (Pure_RNA): {stats['pure_rna']} 个")
    print(f"📁 02_RNA-蛋白质复合体 (RNA_Protein): {stats['rna_protein']} 个")
    print(f"📁 03_游离核糖体 (Ribosome_Apo): {stats['ribosome']} 个")
    print(f"📁 04_核糖体-RNA结合态 (Ribosome_Bound_RNA): {stats['ribosome_bound']} 个")
    print(f"📁 05_其他或失败 (Others_or_Failed): {stats['others']} 个")
    print("\n你可以去 ./pdb_data 目录下查看分类好的文件夹了。")


if __name__ == "__main__":
    classify_cif_files()