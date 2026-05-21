import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

# 设置路径
DATA_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"


def analyze_cif_completeness(filepath):
    """
    分析单个 CIF 文件的链数量和数据完整性
    完整性定义：实际坐标中存在的残基数 / 序列中定义的总残基数
    """
    try:
        # 使用 MMCIF2Dict 快速读取元数据（比解析整个结构快）
        mmcif_dict = MMCIF2Dict(filepath)

        # 1. 获取链的数量
        # _entity_poly.entity_id 通常列出所有的聚合物实体
        if '_entity_poly.entity_id' in mmcif_dict:
            entities = mmcif_dict['_entity_poly.entity_id']
            # 如果只有一个实体，Biopython 返回字符串而非列表，需处理
            num_chains = len(entities) if isinstance(entities, list) else 1
        else:
            num_chains = 0

        # 2. 计算完整性 (基于原子坐标站位的残基)
        # 我们通过对比 _entity_poly_seq (预期序列) 和 _atom_site (实际坐标)
        # 这里使用 MMCIFParser 获取更精确的结构统计
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("RNA", filepath)

        actual_residues = 0
        for model in structure:
            for chain in model:
                actual_residues += len(list(chain.get_residues()))

        # 获取序列预期的长度总和
        expected_residues = 0
        if '_entity_poly.sample_length' in mmcif_dict:
            lengths = mmcif_dict['_entity_poly.sample_length']
            if isinstance(lengths, list):
                expected_residues = sum(int(l) for l in lengths)
            else:
                expected_residues = int(lengths)

        completeness = (actual_residues / expected_residues * 100) if expected_residues > 0 else 0

        return {
            "filename": os.path.basename(filepath),
            "chains": num_chains,
            "actual_res": actual_residues,
            "expected_res": expected_residues,
            "completeness": round(completeness, 2)
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def main():
    results = []
    cif_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.cif')]

    print(f"Found {len(cif_files)} CIF files. Starting analysis...")

    for i, filename in enumerate(cif_files):
        filepath = os.path.join(DATA_DIR, filename)
        data = analyze_cif_completeness(filepath)
        if data:
            results.append(data)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(cif_files)}...")

    df = pd.DataFrame(results)
    df.to_csv("rna_analysis_report.csv", index=False)
    print("Report saved to rna_analysis_report.csv")

    # --- 可视化部分 ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图 1: 链数量分布
    sns.histplot(df['chains'], bins=range(1, df['chains'].max() + 2), ax=axes[0], color='skyblue', kde=False)
    axes[0].set_title('Distribution of Chain Counts')
    axes[0].set_xlabel('Number of Chains per File')
    axes[0].set_ylabel('Frequency')

    # 图 2: 完整性分布
    sns.boxplot(x=df['completeness'], ax=axes[1], color='lightgreen')
    axes[1].set_title('Data Completeness Percentage')
    axes[1].set_xlabel('Completeness (%)')

    plt.tight_layout()
    plt.savefig("rna_data_quality.png")
    print("Plots saved to rna_data_quality.png")


if __name__ == "__main__":
    main()