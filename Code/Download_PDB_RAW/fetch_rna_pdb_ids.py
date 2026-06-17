import requests
import pandas as pd
import time

# ============ 配置 ============
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
OUTPUT_FILE = "./rna_experimental_pdb_ids.xlsx"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 2
# =============================


def _post_query(query):
    """发送查询到 RCSB Search API，返回 PDB ID 集合。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                RCSB_SEARCH_URL,
                json=query,
                timeout=REQUEST_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            result_set = data.get("result_set", [])
            return {entry["identifier"] for entry in result_set}
        except requests.exceptions.RequestException as e:
            print(f"  第 {attempt} 次尝试失败: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise


def search_rna_experimental_structures():
    """
    从 RCSB PDB 检索所有包含 RNA 且有湿实验数据的结构。
    策略：两步查询取差集
      A = 所有包含 RNA 的结构
      B = 包含 RNA 且为纯理论模型的结构
      结果 = A - B
    （RCSB Search API v2 不支持 negation，故用差集代替）
    """
    # 查询 A：所有包含 RNA 的结构
    print("查询 A：获取所有包含 RNA 的结构...")
    query_rna_all = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                "operator": "greater",
                "value": 0,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    set_all = _post_query(query_rna_all)
    print(f"  全量 RNA 结构: {len(set_all)} 个")

    # 查询 B：RNA + 纯理论模型
    print("查询 B：获取 RNA 中的纯理论模型...")
    query_rna_theoretical = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                        "operator": "greater",
                        "value": 0,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "THEORETICAL MODEL",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    set_theoretical = _post_query(query_rna_theoretical)
    print(f"  纯理论模型: {len(set_theoretical)} 个")

    # 差集 = 有实验数据的 RNA 结构
    set_experimental = set_all - set_theoretical
    print(f"差集（有湿实验数据的 RNA 结构）: {len(set_experimental)} 个")

    return sorted(set_experimental)


def save_to_xlsx(pdb_ids, output_path):
    """将 PDB ID 列表写入 xlsx 文件的第一列。"""
    df = pd.DataFrame({"PDB_ID": pdb_ids})
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"已保存 {len(pdb_ids)} 个 PDB ID 至: {output_path}")


def main():
    pdb_ids = search_rna_experimental_structures()
    save_to_xlsx(pdb_ids, OUTPUT_FILE)


if __name__ == "__main__":
    main()
