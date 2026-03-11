import os
import gzip
import shutil
import requests
import pandas as pd
import csv  # 新增：用于逐行写入
from tqdm import tqdm
from Bio.PDB import PDBList, MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionException
import warnings
import time
from functools import wraps

# 忽略无关警告
warnings.simplefilter('ignore', PDBConstructionException)
warnings.filterwarnings('ignore', category=UserWarning)

# 配置项
CONFIG = {
    "save_dir": "./pdb_data",
    "csv_filename": "all_rna_structures.csv",  # 结果文件名
    "rcsb_search_url": "https://search.rcsb.org/rcsbsearch/v2/query",
    "rate_limit_s": 0.5,
    "recommended_min_len": 30,
    "recommended_max_len": 1024,
    "recommended_min_completeness": 0.95,
    "rna_bases": {'A', 'U', 'C', 'G', 'I', 'RA', 'RU', 'RC', 'RG', 'rA', 'rU', 'rC', 'rG',
                  'm6A', 'ψ', '5MU', 'OMU', '假尿苷'},
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
                    time.sleep(delay * (i + 1))
            return None

        return wrapper

    return decorator


class RNADatasetCurator:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir or CONFIG["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.parser = MMCIFParser(QUIET=True)
        self.pdbl = PDBList(verbose=False)

        # CSV文件路径
        self.csv_path = os.path.join(self.save_dir, CONFIG["csv_filename"])

        # 初始化 CSV 表头（如果文件不存在）
        self.columns = ['PDB_ID', 'Title', 'Method', 'Resolution', 'Model_Idx', 'Chain_ID',
                        'Length', 'Completeness', 'Missing_Residues']
        self._init_csv()

    def _init_csv(self):
        """如果文件不存在，写入表头"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()

    def get_processed_ids(self):
        """读取已存在的CSV，获取已经处理过的PDB ID集合"""
        if not os.path.exists(self.csv_path):
            return set()

        processed = set()
        try:
            # 只读取 PDB_ID 列，加快速度
            df = pd.read_csv(self.csv_path, usecols=['PDB_ID'])
            processed = set(df['PDB_ID'].unique())
            print(f"检测到断点文件，已处理 {len(processed)} 个结构。")
        except Exception as e:
            print(f"读取断点文件失败（可能是空文件或格式错误）: {e}")
        return processed

    def save_record_incremental(self, record_list):
        """实时追加写入数据到CSV"""
        if not record_list:
            return

        with open(self.csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            # 过滤掉不在 self.columns 中的多余键，防止报错
            clean_records = []
            for item in record_list:
                clean_item = {k: v for k, v in item.items() if k in self.columns}
                clean_records.append(clean_item)

            writer.writerows(clean_records)

    def search_rna_structures(self):
        """使用 RCSB Search API 查找 RNA 结构"""
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
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result_set = response.json()['result_set']
            id_list = [entry['identifier'] for entry in result_set]
            print(f"从 PDB 找到 {len(id_list)} 个包含 RNA 的结构 ID。")
            return id_list
        except requests.exceptions.RequestException as e:
            print(f"检索ID失败: {e}")
            return []

    def get_metadata_from_cif(self, filepath):
        """鲁棒的元数据提取"""
        try:
            mmcif_dict = MMCIF2Dict(filepath)
            title = mmcif_dict.get('_struct.title', ['Unknown Title'])
            title = "\n".join([t.strip() for t in title]).strip() if isinstance(title, list) else title.strip()

            method = mmcif_dict.get('_exptl.method', ['Unknown'])
            method = ", ".join([m.strip() for m in method]) if isinstance(method, list) else method.strip()

            resolution = mmcif_dict.get('_refine.ls_d_res_high', ['Unknown'])
            resolution = resolution[0] if isinstance(resolution, list) and resolution else 'Unknown'

            return title, method, resolution
        except Exception as e:
            # print(f"提取元数据警告 {filepath}: {e}") # 减少日志输出
            return "Unknown Title", "Unknown", "Unknown"

    def _unzip_gz_file(self, gz_filepath):
        """解压gz文件"""
        cif_filepath = gz_filepath.replace('.gz', '')
        if os.path.exists(cif_filepath) and os.path.getsize(cif_filepath) > 0:
            return cif_filepath

        try:
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(cif_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # 解压成功后删除gz，节省空间
            try:
                os.remove(gz_filepath)
            except:
                pass
            return cif_filepath
        except Exception as e:
            print(f"解压失败 {gz_filepath}: {e}")
            return None

    @retry(max_retries=3, delay=1)
    def download_structure(self, pdb_id):
        """下载结构"""
        pdb_id_lower = pdb_id.lower()
        target_cif = os.path.join(self.save_dir, f"{pdb_id_lower}.cif")

        # 1. 检查已解压的 cif
        if os.path.exists(target_cif) and os.path.getsize(target_cif) > 0:
            return target_cif

        # 2. 检查可能存在的 gz (可能上次下载了没解压)
        target_gz = target_cif + ".gz"
        if os.path.exists(target_gz) and os.path.getsize(target_gz) > 0:
            return self._unzip_gz_file(target_gz)

        # 3. 下载
        try:
            file_path = self.pdbl.retrieve_pdb_file(
                pdb_code=pdb_id_lower,
                pdir=self.save_dir,
                file_format='mmCif',
                overwrite=False
            )
        except Exception as e:
            print(f"下载失败 {pdb_id}: {e}")
            return None

        if not file_path or not os.path.exists(file_path):
            return None

        # 4. 解压
        if file_path.endswith('.gz'):
            return self._unzip_gz_file(file_path)

        return file_path

    def analyze_rna_integrity(self, filepath, pdb_id):
        """分析 RNA 完整性"""
        if not filepath:
            return []

        try:
            # 增加一个超时保护机制的替代方案：
            # 这里不方便加 signal alarm (Windows不支持)，
            # 我们假设如果文件太大 (>100MB)，先跳过分析或者仅仅提取元数据
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if file_size_mb > 200:  # 200MB 以上的文件非常可能是核糖体且会卡死
                print(f"警告: {pdb_id} 文件过大 ({file_size_mb:.2f} MB)，跳过结构分析以防卡死。")
                return []

            structure = self.parser.get_structure(pdb_id, filepath)
        except Exception as e:
            print(f"解析错误 {pdb_id}: {str(e)[:100]}")
            return []

        structure_title, method, resolution = self.get_metadata_from_cif(filepath)
        rna_chains_info = []

        # 优化逻辑：只取第一个 Model (通常是 Model 0)
        # 对于 NMR 即使有多个 Model，只分析第一个可以节省大量时间且对清洗足够
        try:
            model = next(iter(structure))
            model_idx = 0  # 默认为0
        except StopIteration:
            return []

        for chain in model:
            residues = [res for res in chain if res.id[0] == " "]
            if not residues:
                continue

            res_names = [res.get_resname().strip() for res in residues]

            # 快速筛选：必须包含 RNA 碱基
            has_rna = False
            for r in res_names:
                if r in CONFIG["rna_bases"]:
                    has_rna = True
                    break

            if not has_rna:
                continue

            # DNA 排除逻辑
            has_dna_marker = any(d in res_names for d in CONFIG["dna_markers"])
            has_uracil = any('U' in r for r in res_names)

            # 如果含有T但没有U，判定为DNA，跳过
            if has_dna_marker and not has_uracil:
                continue

            # 是 RNA
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
                'Resolution': resolution,
                'Model_Idx': model_idx,
                'Chain_ID': chain.id,
                'Length': seq_len,
                'Completeness': completeness,
                'Missing_Residues': missing_count
            })

        return rna_chains_info

    def run_pipeline(self, max_files=None):
        """主流程：支持断点续传和增量保存"""
        all_ids = self.search_rna_structures()
        if not all_ids:
            return

        # 1. 获取已处理列表
        processed_ids = self.get_processed_ids()

        # 2. 过滤待处理列表
        target_ids = [pid for pid in all_ids if pid not in processed_ids]

        if max_files:
            target_ids = target_ids[:max_files]

        print(f"总共 {len(all_ids)} 个，已处理 {len(processed_ids)} 个，本次计划处理 {len(target_ids)} 个。")

        if not target_ids:
            print("所有任务已完成！")
            return

        # 3. 开始循环
        for pdb_id in tqdm(target_ids, desc="处理进度"):
            # 下载 (内部会自动跳过已存在的有效文件)
            filepath = self.download_structure(pdb_id)
            if not filepath:
                # 下载失败也要记录一下吗？
                # 如果不需要记录失败的，就直接 continue
                # 如果想避免下次卡在这里，可以记录一个空的记录，或者手动加入黑名单
                continue

            # 分析
            chain_data = self.analyze_rna_integrity(filepath, pdb_id)

            # 4. 关键：实时写入 CSV
            if chain_data:
                self.save_record_incremental(chain_data)
            else:
                # 即使没有链数据（比如全DNA或解析失败），
                # 也可以选择是否把这个ID写入另一个 'failed.csv'，
                # 否则下次还会尝试分析它。
                # 为简单起见，这里我们只追加有效数据。
                # ⚠️ 进阶技巧：为了防止反复处理坏文件，我们追加一条只含 PDB_ID 的记录到 CSV
                # 这样下次 get_processed_ids 就能识别它。
                self.save_record_incremental([{'PDB_ID': pdb_id, 'Title': 'No Valid RNA / Parse Error'}])

            # 遵守速率限制
            time.sleep(CONFIG["rate_limit_s"])

        print(f"\n所有处理完成。数据已保存至: {self.csv_path}")

        # 5. 最后生成一个清洗后的推荐列表（读取完整的CSV）
        self.generate_recommendation()

    def generate_recommendation(self):
        """读取生成的全量CSV，筛选出高质量数据"""
        if not os.path.exists(self.csv_path):
            return

        print("正在生成推荐列表...")
        try:
            df = pd.read_csv(self.csv_path)
            # 过滤掉刚才为了占位写入的空行
            df = df[df['Length'].notna()]

            recommended = df[
                (df['Completeness'] >= CONFIG["recommended_min_completeness"]) &
                (df['Length'] >= CONFIG["recommended_min_len"]) &
                (df['Length'] <= CONFIG["recommended_max_len"])
                ].drop_duplicates(subset=['PDB_ID', 'Chain_ID'])

            output_rec = os.path.join(self.save_dir, "recommended_rna_clean.csv")
            recommended.to_csv(output_rec, index=False, encoding='utf-8-sig')
            print(f"高质量数据已筛选: {len(recommended)} 条，保存至 {output_rec}")
        except Exception as e:
            print(f"生成推荐列表出错: {e}")


if __name__ == "__main__":
    curator = RNADatasetCurator()
    # 设置为 None 以跑完剩余所有数据
    curator.run_pipeline(max_files=None)