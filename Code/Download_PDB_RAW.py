import os
import requests
import pandas as pd
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException
import warnings
import time

# 忽略 Biopython 解析非标准 PDB 时的警告
warnings.simplefilter('ignore', PDBConstructionException)


class RNADatasetCurator:
    def __init__(self, save_dir="./pdb_data"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.parser = MMCIFParser(QUIET=True)

    def search_rna_structures(self):
        """
        使用 RCSB Search API 查找 RNA 结构，并同时获取它们的标题。
        """
        print("正在从 PDB 检索 RNA 结构列表 (包含标题)...")
        url = "[https://search.rcsb.org/rcsbsearch/v2/query](https://search.rcsb.org/rcsbsearch/v2/query)"

        # 这是一个 JSON 查询，寻找包含 RNA 聚合物的结构
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                    "operator": "greater",
                    "value": 0
                }
            },
            "return_type": "entry",
            "request_options": {
                "return_all_hits": True
            }
        }

        response = requests.post(url, json=query)
        id_list = []

        if response.status_code == 200:
            result_set = response.json()['result_set']
            id_list = [entry['identifier'] for entry in result_set]
            print(f"找到 {len(id_list)} 个包含 RNA 的结构 ID。")
        else:
            print("检索 ID 失败")
            return {}

        # 进一步：我们需要获取这些 ID 对应的 Title (结构名称)
        # 为了不让 PDB API 崩溃，我们使用 GraphQL 或者简单的分批查询，
        # 但为了脚本简单，这里我们稍后在分析时单独处理，或者只记录 ID。
        # (注：为了工程效率，大规模获取标题建议用 GraphQL，这里为保持脚本简单，
        # 我们将在下载后尝试从文件头读取标题，或者仅列出 ID)

        return id_list

    def get_structure_title_from_cif(self, structure):
        """尝试从解析的结构中获取 Header 标题"""
        # Biopython 的 MMCIFParser 解析后的 header 通常在 structure.header 中
        # 但 MMCIF 格式比较复杂，有时 title 字段位置不固定
        title = structure.header.get('name', 'Unknown Title')
        return title

    def download_structure(self, pdb_id):
        """
        下载 mmCIF 文件
        """
        pdbl = PDBList(verbose=False)
        # retrieve_pdb_file 会自动下载到 self.save_dir
        filepath = pdbl.retrieve_pdb_file(pdb_id, pdir=self.save_dir, file_format='mmCif')
        return filepath

    def analyze_rna_integrity(self, filepath, pdb_id):
        """
        核心功能：检查数据缺失和长度
        """
        try:
            structure = self.parser.get_structure(pdb_id, filepath)
        except Exception as e:
            print(f"解析错误 {pdb_id}: {e}")
            return []

        # 获取结构的标题（名字）
        structure_title = self.get_structure_title_from_cif(structure)

        rna_chains_info = []

        for model in structure:
            for chain in model:
                # 过滤杂原子/水分子
                residues = [res for res in chain if res.id[0] == " "]

                if not residues:
                    continue

                # 简单判断是否为 RNA
                res_names = [res.get_resname().strip() for res in residues]
                # 只要链里包含常见的 RNA 碱基，就认为是 RNA 链
                is_likely_rna = any(n in ['A', 'U', 'C', 'G', 'RA', 'RU', 'RC', 'RG'] for n in res_names)

                if is_likely_rna:
                    seq_len = len(residues)

                    # --- 检查数据缺失 (Missing Residues) ---
                    res_ids = [res.id[1] for res in residues]
                    if len(res_ids) > 1:
                        theoretical_len = max(res_ids) - min(res_ids) + 1
                        missing_count = theoretical_len - seq_len
                        # 覆盖率 = 实际长度 / (实际长度 + 缺失长度)
                        completeness = seq_len / theoretical_len if theoretical_len > 0 else 0
                    else:
                        missing_count = 0
                        completeness = 1.0

                    rna_chains_info.append({
                        'PDB_ID': pdb_id,
                        'Title': structure_title,  # 新增：RNA的名字
                        'Chain_ID': chain.id,
                        'Length': seq_len,
                        'Completeness': round(completeness, 4),  # 完整度 (0-1)
                        'Missing_Residues': missing_count,
                        'Is_Valid': True  # 默认标记，后续可手动改为 False
                    })

            # 只处理第一个 Model (通常是最佳模型)
            break

        return rna_chains_info

    def run_pipeline(self, max_files=None):
        """
        主流程
        :param max_files: 设置为 None 则下载所有数据 (警告：可能很大)
        """
        all_ids = self.search_rna_structures()

        # 如果设置了最大数量（测试用），则切片
        target_ids = all_ids[:max_files] if max_files else all_ids

        results = []

        print(f"计划处理 {len(target_ids)} 个结构...")
        print("注意：全量下载可能需要较长时间和硬盘空间。")

        for index, pdb_id in enumerate(target_ids):
            print(f"[{index + 1}/{len(target_ids)}] Processing {pdb_id}...")

            # 1. 下载
            try:
                filepath = self.download_structure(pdb_id)
            except Exception as e:
                print(f"下载失败 {pdb_id}: {e}")
                continue

            if not os.path.exists(filepath):
                continue

            # 2. 分析 (无论好坏，都收集信息)
            chain_data = self.analyze_rna_integrity(filepath, pdb_id)
            if chain_data:
                results.extend(chain_data)

        # 3. 导出全量报告
        df = pd.DataFrame(results)

        # 调整列的顺序，好看一点
        cols = ['PDB_ID', 'Title', 'Chain_ID', 'Length', 'Completeness', 'Missing_Residues']
        # 确保列存在
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        print("\n=== 分析完成 ===")
        print(f"共统计了 {len(df)} 条 RNA 链的信息。")
        print(df.head())

        # 保存所有数据
        output_file = "all_rna_structures.csv"
        df.to_csv(output_file, index=False)
        print(f"所有数据已保存至: {output_file}")

        # 4. 自动帮你生成一个“推荐列表” (Optional)
        # 这里只是建议，不删除原始数据
        recommended = df[
            (df['Completeness'] > 0.95) &
            (df['Length'] >= 30) &
            (df['Length'] <= 1024)
            ]
        recommended.to_csv("recommended_rna_clean.csv", index=False)
        print(f"已筛选出 {len(recommended)} 条高质量数据保存至: recommended_rna_clean.csv")


if __name__ == "__main__":
    curator = RNADatasetCurator()
    # ⚠️ 注意：如果要下载所有，请把下面这行改成 curator.run_pipeline(max_files=None)
    # 第一次运行建议先用 5 个试试水
    curator.run_pipeline(max_files=5)
