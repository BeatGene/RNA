import os
import json
import re
from collections import defaultdict

# ============ 配置 ============
CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
SIMPLE_JSON_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Simple_json/01_Pure_RNA"
COMPLEX_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"
SEEDS = {"42", "43", "44", "45"}
EXPECTED_SAMPLES = 50
REPORT_FILE = "check_all_report.txt"
# =============================


def get_baseline_pdb_ids(cif_dir):
    """从 01_Pure_RNA 下的 .cif 文件获取基准 PDB ID 列表."""
    if not os.path.exists(cif_dir):
        print(f"[ERROR] CIF 目录不存在: {cif_dir}")
        return []
    ids = sorted([
        f.replace(".cif", "") for f in os.listdir(cif_dir) if f.endswith(".cif")
    ])
    return ids


def check_json_health(filepath):
    """检查 JSON 文件是否可解析且包含基本字段, 返回 (ok, reason)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"JSON 解析失败: {e}"
    except Exception as e:
        return False, f"读取失败: {e}"

    if isinstance(data, list):
        if len(data) == 0:
            return False, "JSON 为空列表"
        data = data[0]

    if "name" not in data:
        return False, "缺少 'name' 字段"
    if "sequences" not in data:
        return False, "缺少 'sequences' 字段"

    return True, "OK"


def check_prep_folder(pdb_id_lower):
    """检查 prep_output_<id> 目录结构是否完整."""
    prep_dir = os.path.join(COMPLEX_DIR, f"prep_output_{pdb_id_lower}")
    if not os.path.isdir(prep_dir):
        return False, "prep_output 文件夹不存在"

    target_dir = os.path.join(prep_dir, pdb_id_lower)
    if not os.path.isdir(target_dir):
        return False, f"缺少子目录 {pdb_id_lower}"

    a3m = os.path.join(target_dir, "rna_msa", "0", "rna_msa.a3m")
    if not os.path.isfile(a3m):
        return False, "缺少 rna_msa/0/rna_msa.a3m"
    if os.path.getsize(a3m) == 0:
        return False, "rna_msa.a3m 为空"

    return True, "OK"


def check_pred_seeds(pdb_id_lower):
    """检查 4 个 seed 的预测输出是否完整 (各有 50 个 CIF)."""
    details = {}
    all_ok = True
    for seed in SEEDS:
        folder_name = f"pred_output_{pdb_id_lower}_seed_{seed}"
        folder_path = os.path.join(COMPLEX_DIR, folder_name)
        if not os.path.isdir(folder_path):
            details[seed] = "文件夹不存在"
            all_ok = False
            continue

        pred_dir = os.path.join(folder_path, pdb_id_lower, f"seed_{seed}", "predictions")
        if not os.path.isdir(pred_dir):
            details[seed] = "缺少 predictions 目录"
            all_ok = False
            continue

        cif_files = [f for f in os.listdir(pred_dir) if f.endswith(".cif")]
        if len(cif_files) != EXPECTED_SAMPLES:
            details[seed] = f"只有 {len(cif_files)} 个 CIF (期望 {EXPECTED_SAMPLES})"
            all_ok = False
        else:
            # 额外检查：CIF 文件不能为空
            empty = [f for f in cif_files if os.path.getsize(os.path.join(pred_dir, f)) == 0]
            if empty:
                details[seed] = f"{len(empty)} 个 CIF 文件为空"
                all_ok = False
            else:
                details[seed] = "OK"
    return all_ok, details


def main():
    baseline = get_baseline_pdb_ids(CIF_DIR)
    total = len(baseline)
    print(f"基准 CIF 文件数: {total}")

    # 分类结果
    all_pass = []           # 5 项全过
    missing_orig_json = []  # 缺原始 JSON
    missing_updt_json = []  # 缺 updated JSON
    bad_json = []           # JSON 存在但不健康
    bad_prep = []           # prep 有问题
    bad_pred = []           # pred 有问题
    bad_pred_detail = {}    # pdb_id -> seed details

    for pdb_id in baseline:
        low = pdb_id.lower()
        issues = []

        # ---- 1. 原始 JSON ----
        orig_json = os.path.join(SIMPLE_JSON_DIR, f"{low}.json")
        if not os.path.isfile(orig_json):
            missing_orig_json.append(pdb_id)
            issues.append("缺原始JSON")
        else:
            ok, reason = check_json_health(orig_json)
            if not ok:
                bad_json.append((pdb_id, "原始JSON", reason))
                issues.append(f"原始JSON不健康: {reason}")

        # ---- 2. Updated JSON ----
        updt_json = os.path.join(SIMPLE_JSON_DIR, f"{low}-final-updated.json")
        if not os.path.isfile(updt_json):
            missing_updt_json.append(pdb_id)
            issues.append("缺updated JSON")
        else:
            ok, reason = check_json_health(updt_json)
            if not ok:
                bad_json.append((pdb_id, "updated JSON", reason))
                issues.append(f"updated JSON不健康: {reason}")

        # ---- 3. Prep 输出 ----
        ok, reason = check_prep_folder(low)
        if not ok:
            bad_prep.append((pdb_id, reason))
            issues.append(f"Prep异常: {reason}")

        # ---- 4. Pred 输出 (4 seeds) ----
        ok, details = check_pred_seeds(low)
        if not ok:
            bad_pred.append(pdb_id)
            bad_pred_detail[pdb_id] = details
            failed_seeds = [s for s in SEEDS if details.get(s) != "OK"]
            issues.append(f"Pred异常 seeds: {failed_seeds}")

        if not issues:
            all_pass.append(pdb_id)

    # ======== 输出报告 ========
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("========== Protenix Prep & Pred 全面检查报告 ==========\n\n")
        f.write(f"基准 CIF 文件数 (01_Pure_RNA): {total}\n")
        f.write(f"全部通过 (5/5):                {len(all_pass)}\n")
        f.write(f"缺原始 JSON:                   {len(missing_orig_json)}\n")
        f.write(f"缺 updated JSON:               {len(missing_updt_json)}\n")
        f.write(f"JSON 不健康:                   {len(bad_json)}\n")
        f.write(f"Prep 异常:                     {len(bad_prep)}\n")
        f.write(f"Pred 异常:                     {len(bad_pred)}\n")
        f.write(f"总通过率: {len(all_pass)}/{total} = {len(all_pass)/total*100:.1f}%\n")

        if missing_orig_json:
            f.write(f"\n--- 缺原始 JSON ({len(missing_orig_json)} 个) ---\n")
            for pid in missing_orig_json:
                f.write(f"  {pid}\n")

        if missing_updt_json:
            f.write(f"\n--- 缺 updated JSON ({len(missing_updt_json)} 个) ---\n")
            for pid in missing_updt_json:
                f.write(f"  {pid}\n")

        if bad_json:
            f.write(f"\n--- JSON 不健康 ({len(bad_json)} 个) ---\n")
            for pid, jtype, reason in bad_json:
                f.write(f"  {pid} [{jtype}]: {reason}\n")

        if bad_prep:
            f.write(f"\n--- Prep 异常 ({len(bad_prep)} 个) ---\n")
            for pid, reason in bad_prep:
                f.write(f"  {pid}: {reason}\n")

        if bad_pred:
            f.write(f"\n--- Pred 异常 ({len(bad_pred)} 个) ---\n")
            for pid in sorted(bad_pred):
                f.write(f"  {pid}:\n")
                for seed in sorted(SEEDS):
                    status = bad_pred_detail.get(pid, {}).get(seed, "?")
                    if status != "OK":
                        f.write(f"    seed_{seed}: {status}\n")

        if all_pass:
            f.write(f"\n--- 全部通过 ({len(all_pass)} 个) ---\n")
            for pid in all_pass:
                f.write(f"  {pid}\n")

    # 终端汇总
    print(f"\n{'='*50}")
    print(f"全部通过: {len(all_pass)}/{total} ({len(all_pass)/total*100:.1f}%)")
    print(f"缺原始JSON: {len(missing_orig_json)}")
    print(f"缺updated JSON: {len(missing_updt_json)}")
    print(f"JSON不健康: {len(bad_json)}")
    print(f"Prep异常: {len(bad_prep)}")
    print(f"Pred异常: {len(bad_pred)}")
    print(f"详细报告已保存至: {REPORT_FILE}")


if __name__ == "__main__":
    main()
