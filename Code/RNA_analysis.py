import os
import pandas as pd
from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from tqdm import tqdm
import warnings
from Bio import BiopythonWarning

# 忽略 Biopython 解析时的一些无害警告
warnings.simplefilter('ignore', BiopythonWarning)


def _get_cif_first(cif_dict, key, default="Unknown"):
    """安全获取 CIF 字典的第一个值，兼容字符串和列表"""
    val = cif_dict.get(key, default)
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    return val


def _ensure_list(val):
    """安全地将 CIF 字典的返回值转换为列表，防止单行数据被解析为字符串导致遍历错误"""
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    return val


def parse_rna_cif_directory(cif_dir, output_csv):
    """
    遍历指定目录下的所有 CIF 文件，提取结构信息并严谨评估缺失率
    """
    parser = FastMMCIFParser(QUIET=True)
    all_chain_data = []

    # 获取所有 cif 文件
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif') or f.endswith('.cif.gz')]
    total_files = len(cif_files)
    print(f"[*] 发现纯 RNA CIF 文件总数: {total_files}")

    error_count = 0

    for filename in tqdm(cif_files, desc="正在解析 CIF 文件"):
        filepath = os.path.join(cif_dir, filename)
        pdb_id = filename.split('.')[0][:4].upper()

        try:
            # 1. 使用字典模式快速读取 Header 信息
            cif_dict = MMCIF2Dict(filepath)

            # 【修复 1】安全提取基础信息与分辨率清理
            title = _get_cif_first(cif_dict, '_struct.title', 'Unknown')
            method = _get_cif_first(cif_dict, '_exptl.method', 'Unknown')

            resolution = _get_cif_first(cif_dict, '_refine.ls_d_res_high',
                                        _get_cif_first(cif_dict, '_reflns.d_resolution_high', 'N/A'))
            if resolution in ['?', '.', 'None']:
                resolution = 'N/A'

            # 2. 提取序列真实长度字典 (从 _entity_poly_seq 表)
            seq_length_dict = {}
            if '_entity_poly_seq.entity_id' in cif_dict and '_entity_poly_seq.num' in cif_dict:
                ent_ids = _ensure_list(cif_dict['_entity_poly_seq.entity_id'])
                res_nums = _ensure_list(cif_dict['_entity_poly_seq.num'])
                for ent_id, res_num in zip(ent_ids, res_nums):
                    try:
                        seq_length_dict[ent_id] = max(seq_length_dict.get(ent_id, 0), int(res_num))
                    except ValueError:
                        pass

            # 【修复 2 & 3 前置准备】构建 ID 映射关系与 Chain->Entity 映射
            asym_id_map = {}
            chain_to_entity = {}

            if '_struct_asym.id' in cif_dict:
                label_ids = _ensure_list(cif_dict['_struct_asym.id'])
                auth_ids = _ensure_list(cif_dict.get('_struct_asym.auth_asym_id', label_ids))
                entity_ids = _ensure_list(cif_dict.get('_struct_asym.entity_id', label_ids))

                for lbl, auth, ent in zip(label_ids, auth_ids, entity_ids):
                    asym_id_map[lbl] = auth
                    chain_to_entity[lbl] = ent

            # 3. 解析真实 3D 坐标结构
            structure = parser.get_structure(pdb_id, filepath)

            # 遍历模型和链
            for model in structure:
                model_idx = model.id
                total_chains_in_model = len(model.child_list)

                for chain in model:
                    chain_label_id = chain.id
                    # 【修复 2 核心】使用真实的 auth_asym_id
                    chain_auth_id = asym_id_map.get(chain_label_id, chain_label_id)

                    # 实际解析出的残基数
                    resolved_residues = [res for res in chain if res.id[0] == ' ']
                    resolved_count = len(resolved_residues)

                    # 统计 missing_count (使用 auth_asym_id 进行精确匹配)
                    missing_count = 0
                    if '_pdbx_unobs_or_zero_occ_residues.auth_asym_id' in cif_dict:
                        missing_res_chains = _ensure_list(cif_dict['_pdbx_unobs_or_zero_occ_residues.auth_asym_id'])
                        missing_count = missing_res_chains.count(chain_auth_id)

                    # 【修复 3 核心】优先使用 seq_length_dict 获取真实实验长度
                    current_entity_id = chain_to_entity.get(chain_label_id)
                    if current_entity_id and current_entity_id in seq_length_dict:
                        chain_seq_length = seq_length_dict[current_entity_id]
                    else:
                        # 兜底方案
                        chain_seq_length = resolved_count + missing_count

                    # 计算完整度
                    completeness = resolved_count / chain_seq_length if chain_seq_length > 0 else 0

                    # 保存该链的数据记录
                    chain_record = {
                        'PDB_ID': pdb_id,
                        'Title': title,
                        'Method': method,
                        'Resolution': resolution,
                        'Model_Idx': model_idx,
                        'Chain_ID': chain_auth_id,  # 输出给大家看熟悉的 auth ID
                        'Total_Chains_In_File': total_chains_in_model,
                        'Seq_Length': chain_seq_length,
                        'Resolved_Count': resolved_count,
                        'Missing_Residues': missing_count,
                        'Completeness': round(completeness, 4)
                    }
                    all_chain_data.append(chain_record)

        except Exception as e:
            error_count += 1
            # print(f"Error parsing {filename}: {e}")
            pass

    # 汇总输出
    df = pd.DataFrame(all_chain_data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 40)
    print(f"[*] 解析完成！成功提取 {len(df)} 条 RNA 链的数据。")
    print(f"[*] 解析失败的文件数: {error_count}")
    print(f"[*] 结果已保存至: {output_csv}")

    # 多聚体分布统计
    if not df.empty:
        print("\n[*] 纯 RNA 文件链数分布 (帮你判断多少文件包含多条链):")
        chain_distribution = df.drop_duplicates(subset=['PDB_ID', 'Model_Idx'])['Total_Chains_In_File'].value_counts()
        print(chain_distribution.sort_index())
    print("=" * 40)


if __name__ == '__main__':
    # 【注意】请将下面的路径替换为你服务器上纯RNA文件夹的实际路径
    CIF_DIRECTORY = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
    OUTPUT_FILE = "./pure_rna_analysis_results.csv"

    if os.path.exists(CIF_DIRECTORY):
        parse_rna_cif_directory(CIF_DIRECTORY, OUTPUT_FILE)
    else:
        print(f"[!] 错误: 找不到目录 {CIF_DIRECTORY}，请修改代码中的路径。")