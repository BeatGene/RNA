import os
import glob

PREP_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"
OUTPUT_FILE = "prep_check_result.txt"

def check_prep_folder(folder_path, target):
    """
    检查 prep_output_<target> 内部是否完整。
    返回: (is_ok, reason)
    """
    # 1. <target>/ 子目录
    target_dir = os.path.join(folder_path, target)
    if not os.path.isdir(target_dir):
        return False, f"缺少子目录 {target}"

    # 2. <target>/rna_msa/
    msa_dir = os.path.join(target_dir, "rna_msa")
    if not os.path.isdir(msa_dir):
        return False, "缺少 rna_msa 目录"

    # 3. <target>/rna_msa/0/
    zero_dir = os.path.join(msa_dir, "0")
    if not os.path.isdir(zero_dir):
        return False, "缺少 rna_msa/0 目录"

    # 4. <target>/rna_msa/0/rna_msa.a3m 存在且非空
    a3m_file = os.path.join(zero_dir, "rna_msa.a3m")
    if not os.path.isfile(a3m_file):
        return False, "缺少 rna_msa/0/rna_msa.a3m 文件"
    if os.path.getsize(a3m_file) == 0:
        return False, "rna_msa.a3m 文件为空"

    return True, "OK"

def main():
    if not os.path.exists(PREP_DIR):
        print(f"错误：目录不存在 {PREP_DIR}")
        return

    # 找出所有 prep_output_* 文件夹
    all_folders = [d for d in os.listdir(PREP_DIR)
                   if d.startswith("prep_output_") and os.path.isdir(os.path.join(PREP_DIR, d))]

    normal = []
    abnormal = []

    for folder in sorted(all_folders):
        target = folder.replace("prep_output_", "")
        folder_path = os.path.join(PREP_DIR, folder)
        ok, reason = check_prep_folder(folder_path, target)
        if ok:
            normal.append(target)
        else:
            abnormal.append((target, reason))

    # 生成报告
    with open(OUTPUT_FILE, 'w') as f:
        f.write("========== prep_output 检查报告 ==========\n")
        f.write(f"总 prep_output 数量: {len(all_folders)}\n")
        f.write(f"正常: {len(normal)}\n")
        f.write(f"异常: {len(abnormal)}\n\n")
        if abnormal:
            f.write("---------- 异常详情 ----------\n")
            for tgt, reason in abnormal:
                f.write(f"{tgt}: {reason}\n")
        f.write("\n---------- 正常列表 ----------\n")
        for tgt in normal:
            f.write(f"{tgt}\n")

    print(f"检查完成。报告保存于：{OUTPUT_FILE}")
    print(f"正常 {len(normal)}，异常 {len(abnormal)}")

if __name__ == "__main__":
    main()