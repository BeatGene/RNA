import os
import gzip
import shutil
import requests
import pandas as pd
from tqdm import tqdm  # 新增：进度条
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionException
import warnings
import time
from functools import wraps

# 忽略无关警告
warnings.simplefilter('ignore', PDBConstructionException)
warnings.filterwarnings('ignore', category=UserWarning)

# 配置项（集中管理，便于修改）
CONFIG = {
    "save_dir": "./pdb_data",
    "rcsb_search_url": "https://search.rcsb.org/rcsbsearch/v2/query",
    "rate_limit_s": 1.0,  # RCSB建议的请求间隔
    "recommended_min_len": 30,
    "recommended_max_len": 1024,
    "recommended_min_completeness": 0.95,
    "rna_bases": {'A', 'U', 'C', 'G', 'I', 'RA', 'RU', 'RC', 'RG', 'rA', 'rU', 'rC', 'rG',
                  'm6A', 'ψ', '5MU', 'OMU', '假尿苷'},  # 补充修饰碱基
    "dna_markers": {'T', 'DT', 'dT'}
}

# 重试装饰器
def retry(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == max_retries - 1:
                        print(f"重试{max_retries}次失败: {args}, 错误: {e}")
                        return None
                    time.sleep(delay * (i + 1))  # 指数退避
            return None
        return wrapper
    return decorator

class RNADatasetCurator:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir or CONFIG["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.parser = MMCIFParser(QUIET=True)
        # 初始化PDBList（适配新版本）
        self.pdbl = PDBList(verbose=False)

    def search_rna_structures(self):
        """使用 RCSB Search API 查找 RNA 结构（修复URL+增强错误处理）"""
        print("正在从 PDB 检索 RNA 结构列表...")
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

        try:
            response = requests.post(
                CONFIG["rcsb_search_url"],
                json=query,
                timeout=30,
                headers={"Content-Type": "application/json"}  # 新增：指定请求头
            )
            response.raise_for_status()  # 抛出HTTP错误
            result_set = response.json()['result_set']
            id_list = [entry['identifier'] for entry in result_set]
            print(f"找到 {len(id_list)} 个包含 RNA 的结构 ID。")
            return id_list
        except requests.exceptions.RequestException as e:
            print(f"检索ID失败: {e}")
            return []

    def get_metadata_from_cif(self, filepath):
        """增强：更鲁棒的元数据提取"""
        try:
            mmcif_dict = MMCIF2Dict(filepath)
            # 标题处理：保留换行
            title = mmcif_dict.get('_struct.title', ['Unknown Title'])
            title = "\n".join([t.strip() for t in title]).strip() if isinstance(title, list) else title.strip()
            # 实验方法
            method = mmcif_dict.get('_exptl.method', ['Unknown'])
            method = ", ".join([m.strip() for m in method]) if isinstance(method, list) else method.strip()
            # 新增：分辨率（RNA结构重要指标）
            resolution = mmcif_dict.get('_refine.ls_d_res_high', ['Unknown'])
            resolution = resolution[0] if isinstance(resolution, list) and resolution else 'Unknown'
            return title, method, resolution
        except Exception as e:
            print(f"提取元数据失败 {filepath}: {e}")
            return "Unknown Title", "Unknown", "Unknown"

    def _unzip_gz_file(self, gz_filepath):
        """增强：解压+校验+清理压缩包"""
        cif_filepath = gz_filepath.replace('.gz', '')
        if os.path.exists(cif_filepath):
            # 校验文件是否损坏（简单大小检查）
            if os.path.getsize(cif_filepath) > 0:
                return cif_filepath
            else:
                os.remove(cif_filepath)  # 删除损坏文件

        try:
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(cif_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_filepath)  # 解压后删除压缩包
            return cif_filepath
        except Exception as e:
            print(f"解压失败 {gz_filepath}: {e}")
            return None

    @retry(max_retries=3, delay=1)  # 新增：重试机制
    def download_structure(self, pdb_id):
        """增强：适配Biopython版本+重试+校验"""
        pdb_id_lower = pdb_id.lower()
        target_cif = os.path.join(self.save_dir, f"{pdb_id_lower}.cif")

        # 检查目标文件是否存在且有效
        if os.path.exists(target_cif) and os.path.getsize(target_cif) > 0:
            return target_cif

        try:
            # 适配Biopython新版本：file_format='cif'
            file_path = self.pdbl.retrieve_pdb_file(
                pdb_code=pdb_id_lower,
                pdir=self.save_dir,
                file_format='mmCif',  # 新版本参数
                overwrite=False
            )
        except Exception as e:
            print(f"下载失败 {pdb_id}: {e}")
            return None

        if not file_path or not os.path.exists(file_path):
            return None

        # 处理压缩包
        if file_path.endswith('.gz'):
            cif_filepath = self._unzip_gz_file(file_path)
        else:
            cif_filepath = file_path

        # 最终校验
        if cif_filepath and os.path.exists(cif_filepath) and os.path.getsize(cif_filepath) > 0:
            return cif_filepath
        else:
            return None

    def analyze_rna_integrity(self, filepath, pdb_id):
        """增强：优化RNA判定逻辑+保留多Model信息+完善错误处理"""
        if not filepath:
            return []

        try:
            structure = self.parser.get_structure(pdb_id, filepath)
        except Exception as e:
            print(f"解析错误 {pdb_id}: {str(e)}")
            return []

        # 获取元数据
        structure_title, method, resolution = self.get_metadata_from_cif(filepath)
        rna_chains_info = []

        for model_idx, model in enumerate(structure):
            for chain in model:
                # 过滤杂原子（只保留标准残基）
                residues = [res for res in chain if res.id[0] == " "]
                if not residues:
                    continue

                res_names = [res.get_resname().strip() for res in residues]
                # 优化RNA判定逻辑
                has_rna_base = any(any(base in res for base in CONFIG["rna_bases"]) for res in res_names)
                has_dna_base = any(any(marker in res for marker in CONFIG["dna_markers"]) for res in res_names)
                has_uracil = any('U' in res for res in res_names)

                # 判定规则：优先基于RNA特征，DNA特征仅作为排除项
                is_rna = has_rna_base and not (has_dna_base and not has_uracil)

                if is_rna:
                    seq_len = len(residues)
                    res_ids = [res.id[1] for res in residues]

                    if len(res_ids) > 1:
                        min_id, max_id = min(res_ids), max(res_ids)
                        theoretical_len = max_id - min_id + 1
                        missing_count = theoretical_len - seq_len
                        completeness = round(seq_len / theoretical_len, 4) if theoretical_len > 0 else 0.0
                    else:
                        missing_count = 0
                        completeness = 1.0

                    rna_chains_info.append({
                        'PDB_ID': pdb_id,
                        'Title': structure_title,
                        'Method': method,
                        'Resolution': resolution,  # 新增
                        'Model_Idx': model_idx,    # 新增：保留Model索引
                        'Chain_ID': chain.id,
                        'Length': seq_len,
                        'Completeness': completeness,
                        'Missing_Residues': missing_count
                    })

        return rna_chains_info

    def run_pipeline(self, max_files=None):
        """主流程：新增进度条+参数化筛选+统一输出路径"""
        all_ids = self.search_rna_structures()
        if not all_ids:
            print("未找到任何RNA结构ID")
            return

        target_ids = all_ids[:max_files] if max_files else all_ids
        results = []

        print(f"\n开始处理 {len(target_ids)} 个结构...")
        # 新增：进度条
        for pdb_id in tqdm(target_ids, desc="处理进度"):
            # 下载结构
            filepath = self.download_structure(pdb_id)
            if not filepath:
                continue

            # 分析RNA完整性
            chain_data = self.analyze_rna_integrity(filepath, pdb_id)
            if chain_data:
                results.extend(chain_data)

            # 遵守速率限制
            time.sleep(CONFIG["rate_limit_s"])

        if not results:
            print("未解析到任何RNA链数据")
            return

        # 整理数据
        df = pd.DataFrame(results)
        # 确保列顺序
        cols = ['PDB_ID', 'Title', 'Method', 'Resolution', 'Model_Idx', 'Chain_ID',
                'Length', 'Completeness', 'Missing_Residues']
        df = df[[c for c in cols if c in df.columns]]

        # 统一输出路径到save_dir
        output_all = os.path.join(self.save_dir, "all_rna_structures.csv")
        df.to_csv(output_all, index=False, encoding='utf-8-sig')
        print(f"\n所有RNA数据已保存至: {output_all}")
        print(f"共解析到 {len(df)} 条RNA链")

        # 推荐列表（参数化筛选）
        recommended = df[
            (df['Completeness'] >= CONFIG["recommended_min_completeness"]) &
            (df['Length'] >= CONFIG["recommended_min_len"]) &
            (df['Length'] <= CONFIG["recommended_max_len"])
        ].drop_duplicates(subset=['PDB_ID', 'Chain_ID'])

        output_recommended = os.path.join(self.save_dir, "recommended_rna_clean.csv")
        recommended.to_csv(output_recommended, index=False, encoding='utf-8-sig')
        print(f"高完整性RNA列表已保存至: {output_recommended}")
        print(f"推荐列表共 {len(recommended)} 条RNA链")


if __name__ == "__main__":
    # 安装依赖（首次运行）
    # pip install biopython pandas requests tqdm
    curator = RNADatasetCurator()
    # 测试时建议设置max_files=10，全量运行设为None
    curator.run_pipeline(max_files=None)