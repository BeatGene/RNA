import os
import gzip
import shutil
import requests
import pandas as pd
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException
import warnings
import time

# 忽略 Biopython 解析非标准 PDB 时的警告
warnings.simplefilter('ignore', PDBConstructionException)
# 忽略pandas的无关警告
warnings.filterwarnings('ignore', category=UserWarning)


class RNADatasetCurator:
    def __init__(self, save_dir="./pdb_data"):
        self.save_dir = save_dir
        # 优化：exist_ok=True 一行替代if判断，更简洁
        os.makedirs(save_dir, exist_ok=True)
        self.parser = MMCIFParser(QUIET=True)

    def search_rna_structures(self):
        """
        使用 RCSB Search API 查找 RNA 结构，并同时获取它们的标题。
        """
        print("正在从 PDB 检索 RNA 结构列表 (包含标题)...")
        # ✅ 修复1：去掉多余的[]和()，修正URL语法错误
        url = "https://search.rcsb.org/rcsbsearch/v2/query"

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

        # ✅ 修复6：增加timeout=30，防止请求卡死
        response = requests.post(url, json=query, timeout=30)
        id_list = []

        if response.status_code == 200:
            result_set = response.json()['result_set']
            id_list = [entry['identifier'] for entry in result_set]
            print(f"找到 {len(id_list)} 个包含 RNA 的结构 ID。")
        else:
            print(f"检索 ID 失败，响应码: {response.status_code}")
            return []

        return id_list

    def get_structure_title_from_cif(self, structure):
        """尝试从解析的结构中获取 Header 标题"""
        # ✅ 修复4：修改为正确的标题key，获取真实的PDB结构标题
        title = structure.header.get('structure_title', 'Unknown Title')
        return title

    def _unzip_gz_file(self, gz_filepath):
        """内部方法：解压gz压缩包，返回解压后的cif文件路径"""
        cif_filepath = gz_filepath.replace('.gz', '')
        if not os.path.exists(cif_filepath):
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(cif_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return cif_filepath

    def download_structure(self, pdb_id):
        """
        下载 mmCIF 文件 + 自动解压.gz压缩包
        """
        pdbl = PDBList(verbose=False)
        pdb_id_lower = pdb_id.lower()
        # 拼接解压后的文件路径，用于判断是否已下载
        target_cif = os.path.join(self.save_dir, f"{pdb_id_lower}.cif")

        # ✅ 修复8：文件已存在则跳过下载，直接返回路径
        if os.path.exists(target_cif):
            return target_cif

        # ✅ 修复2：Biopython新版本参数修正 file_format→format，mmCif→mmcif
        gz_filepath = pdbl.retrieve_pdb_file(pdb_id_lower, pdir=self.save_dir, format='mmcif')
        if not gz_filepath or not os.path.exists(gz_filepath):
            return None

        # ✅ 修复3：自动解压gz压缩包，返回可解析的cif文件路径
        cif_filepath = self._unzip_gz_file(gz_filepath)
        return cif_filepath

    def analyze_rna_integrity(self, filepath, pdb_id):
        """
        核心功能：检查数据缺失和长度
        """
        try:
            structure = self.parser.get_structure(pdb_id, filepath)
        except Exception as e:
            print(f"解析错误 {pdb_id}: {str(e)[:50]}...")
            return []

        structure_title = self.get_structure_title_from_cif(structure)
        rna_chains_info = []

        for model in structure:
            for chain in model:
                # 过滤杂原子/水分子/配体，只保留主链残基
                residues = [res for res in chain if res.id[0] == " "]
                if not residues:
                    continue

                # 简单判断是否为 RNA
                res_names = [res.get_resname().strip() for res in residues]
                # ✅ 修复5：补充小写r开头的残基名，解决RNA链漏检问题
                rna_res_types = ['A', 'U', 'C', 'G', 'RA', 'RU', 'RC', 'RG', 'rA', 'rU', 'rC', 'rG']
                is_likely_rna = any(n in rna_res_types for n in res_names)

                if is_likely_rna:
                    seq_len = len(residues)
                    res_ids = [res.id[1] for res in residues]

                    if len(res_ids) > 1:
                        min_id, max_id = min(res_ids), max(res_ids)
                        theoretical_len = max_id - min_id + 1
                        missing_count = theoretical_len - seq_len
                        completeness = seq_len / theoretical_len if theoretical_len > 0 else 0.0
                    else:
                        missing_count = 0
                        completeness = 1.0

                    rna_chains_info.append({
                        'PDB_ID': pdb_id,
                        'Title': structure_title,
                        'Chain_ID': chain.id,
                        'Length': seq_len,
                        'Completeness': round(completeness, 4),
                        'Missing_Residues': missing_count,
                        'Is_Valid': True
                    })
            # 只处理第一个 Model (通常是最佳模型)
            break

        return rna_chains_info

    def run_pipeline(self, max_files=None):
        """
        主流程
        :param max_files: 设置为 None 则下载所有数据
        """
        all_ids = self.search_rna_structures()
        if not all_ids:
            print("未检索到任何RNA结构ID，程序退出")
            return

        target_ids = all_ids[:max_files] if max_files else all_ids
        results = []

        print(f"计划处理 {len(target_ids)} 个结构...")
        print("注意：全量下载可能需要较长时间和硬盘空间。")

        for index, pdb_id in enumerate(target_ids):
            print(f"[{index + 1}/{len(target_ids)}] Processing {pdb_id}...")

            # 1. 下载+解压
            try:
                filepath = self.download_structure(pdb_id)
            except Exception as e:
                print(f"下载失败 {pdb_id}: {e}")
                continue

            if not os.path.exists(filepath):
                continue

            # 2. 分析
            chain_data = self.analyze_rna_integrity(filepath, pdb_id)
            if chain_data:
                results.extend(chain_data)

            # ✅ 修复7：增加下载间隔，规避RCSB的IP封禁风控
            time.sleep(0.5)

        if not results:
            print("未解析到任何有效RNA链数据")
            return

        # 3. 导出全量报告
        df = pd.DataFrame(results)
        cols = ['PDB_ID', 'Title', 'Chain_ID', 'Length', 'Completeness', 'Missing_Residues']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        print("\n=== 分析完成 ===")
        print(f"共统计了 {len(df)} 条 RNA 链的信息。")
        print(df.head())

        output_file = "all_rna_structures.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"所有数据已保存至: {output_file}")

        # 4. 生成高质量推荐列表
        recommended = df[
            (df['Completeness'] > 0.95) &
            (df['Length'] >= 30) &
            (df['Length'] <= 1024)
            ].drop_duplicates(subset=['PDB_ID', 'Chain_ID'])

        recommended.to_csv("recommended_rna_clean.csv", index=False, encoding='utf-8-sig')
        print(f"已筛选出 {len(recommended)} 条高质量数据保存至: recommended_rna_clean.csv")


if __name__ == "__main__":
    curator = RNADatasetCurator()
    # ⚠️ 注意：如果要下载所有，请把下面这行改成 curator.run_pipeline(max_files=None)
    # 第一次运行建议先用 5 个试试水
    curator.run_pipeline(max_files=1)