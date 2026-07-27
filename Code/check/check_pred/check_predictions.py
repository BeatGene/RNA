import os
import re
from collections import defaultdict

# ========== 配置 ==========
PRED_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"
TARGET_SEEDS = {'42', '43', '44', '45'}   # 要检查的种子
REQUIRED_NUM = 50                            # 每个种子期望的 CIF 文件数

# ========== 工具函数 ==========
def check_target(target, base_dir):
    """
    检查一个目标的四个种子是否完整。
    返回: (normal: bool, details: dict)
    details: 种子 -> 状态/文件数
    """
    seeds_status = {}
    all_ok = True
    for seed in TARGET_SEEDS:
        folder_name = f"pred_output_{target}_seed_{seed}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            seeds_status[seed] = "missing folder"
            all_ok = False
            continue

        # 检查内部结构: <target>/seed_<seed>/predictions/
        pred_dir = os.path.join(folder_path, target, f"seed_{seed}", "predictions")
        if not os.path.isdir(pred_dir):
            seeds_status[seed] = "missing predictions dir"
            all_ok = False
            continue

        # 统计 CIF 文件
        cif_files = [f for f in os.listdir(pred_dir) if f.endswith('.cif')]
        if len(cif_files) != REQUIRED_NUM:
            seeds_status[seed] = f"{len(cif_files)} files (expected {REQUIRED_NUM})"
            all_ok = False
        else:
            seeds_status[seed] = "OK"
    return all_ok, seeds_status

# ========== 主逻辑 ==========
def main():
    # 1. 找出所有符合格式的文件夹，提取目标名
    pattern = re.compile(r'^pred_output_(.+)_seed_(\d+)$')
    target_sets = defaultdict(set)   # target -> set of seeds 实际存在的种子

    for item in os.listdir(PRED_DIR):
        item_path = os.path.join(PRED_DIR, item)
        if not os.path.isdir(item_path):
            continue
        m = pattern.match(item)
        if not m:
            # 忽略如 pred_output_8d2a 这种废文件夹
            continue
        target = m.group(1)
        seed = m.group(2)
        # 只关注我们关心的种子
        if seed not in TARGET_SEEDS:
            continue
        target_sets[target].add(seed)

    if not target_sets:
        print("未找到任何符合格式的采样输出文件夹。")
        return

    # 2. 对每个目标进行检查
    normal_targets = []
    abnormal_targets = []

    for target in sorted(target_sets.keys()):
        ok, details = check_target(target, PRED_DIR)
        if ok:
            normal_targets.append(target)
        else:
            abnormal_targets.append(target)
            print(f"[异常] {target}:")
            for seed in sorted(TARGET_SEEDS):
                status = details.get(seed, "missing folder")
                if status != "OK":
                    print(f"  seed_{seed}: {status}")

    # 3. 打印汇总
    print("\n========== 检查汇总 ==========")
    print(f"总目标数: {len(target_sets)}")
    print(f"完全正常: {len(normal_targets)}")
    print(f"存在异常: {len(abnormal_targets)}")
    if normal_targets:
        print(f"正常目标列表: {', '.join(normal_targets)}")
    if abnormal_targets:
        print(f"异常目标列表: {', '.join(abnormal_targets)}")

if __name__ == "__main__":
    main()