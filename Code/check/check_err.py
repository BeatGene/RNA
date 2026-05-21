import os
import re
from collections import defaultdict

# ================= 配置 =================
BASE_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"
SUBDIRS = ["train", "test", "val"]
SEEDS = ['42', '43', '44', '45']
OUTPUT_FILE = "pred_check_result.txt"

# ================= 工具函数 =================
def collect_targets():
    """
    遍历 train/test/val，收集所有目标 ID（.cif 文件名去掉扩展名）。
    返回:
        targets: dict {id: [subdir1, subdir2, ...]}  一个目标可能出现在多个 split
        total_targets: 不重复的目标数量
    """
    targets = defaultdict(list)
    for sub in SUBDIRS:
        sub_path = os.path.join(BASE_DIR, sub)
        if not os.path.exists(sub_path):
            print(f"警告: 目录不存在 {sub_path}")
            continue
        for item in os.listdir(sub_path):
            item_path = os.path.join(sub_path, item)
            if os.path.isdir(item_path):
                # 查找里面的 .cif 文件
                cif_files = [f for f in os.listdir(item_path) if f.endswith('.cif')]
                if cif_files:
                    # 目标 ID 就是 cif 文件名（不含扩展名）或文件夹名
                    # 按用户描述，文件夹名就是 cif 标号，例如 9j6p
                    target_id = item  # 直接用文件夹名作为 ID
                    targets[target_id].append(sub)
    return targets

def check_pred_status(target_id, subdir):
    """
    检查某个目标在指定数据集（train/test/val）下的四个种子预测状态。
    返回: dict {seed: status_string}
        status_string: "成功" 或 "失败: <错误简述>"
    """
    status = {}
    sample_dir = os.path.join(BASE_DIR, subdir, target_id, "sample")
    if not os.path.exists(sample_dir):
        for seed in SEEDS:
            status[seed] = "失败: sample 文件夹不存在"
        return status

    for seed in SEEDS:
        pred_dir = os.path.join(sample_dir, f"pred_output_{target_id}_seed_{seed}")
        if not os.path.exists(pred_dir):
            status[seed] = "失败: 预测文件夹不存在"
            continue
        err_path = os.path.join(pred_dir, "ERR", "error.txt")
        if os.path.exists(err_path):
            try:
                with open(err_path, 'r') as f:
                    content = f.read().strip()
                # 提取第一行作为简述
                first_line = content.split('\n')[0] if content else "空错误文件"
                # 如果内容太长，截断
                if len(first_line) > 120:
                    first_line = first_line[:120] + "..."
                status[seed] = f"失败: {first_line}"
            except:
                status[seed] = "失败: 无法读取错误文件"
        else:
            # 没有 ERR 目录，假定成功（可根据需要进一步检查 predictions 目录）
            status[seed] = "成功"
    return status

def classify_error(error_msg):
    """简单的错误分类，返回一个标签"""
    if "Dataloader initialization failed" in error_msg and "mmcif directory" in error_msg:
        return "缺少 mmcif 模板数据"
    if "use_template=false" in error_msg:
        return "需设置 use_template=false"
    if "CUDA out of memory" in error_msg or "OOM" in error_msg:
        return "显存不足(OOM)"
    if "FileNotFoundError" in error_msg or "No such file" in error_msg:
        return "输入文件缺失"
    return "其他错误"

# ================= 主流程 =================
def main():
    # 1. 收集目标信息
    print("正在收集目标信息...")
    targets = collect_targets()
    unique_ids = sorted(targets.keys())
    total_unique = len(unique_ids)

    # 2. 检查重复
    duplicates = {tid: splits for tid, splits in targets.items() if len(splits) > 1}
    duplicate_flag = len(duplicates) > 0

    # 3. 遍历检查预测状态
    print("正在检查预测状态（这可能需要一些时间）...")
    result_lines = []
    error_counter = defaultdict(int)  # 错误分类计数
    failed_targets = 0
    total_seeds_checked = 0
    success_seeds = 0
    failed_seeds = 0

    # 记录每个目标的汇总
    for target_id in unique_ids:
        # 该目标可能出现在多个 split，一般只在一个split，但以防万一，取第一个
        sub = targets[target_id][0]
        status_dict = check_pred_status(target_id, sub)

        target_failed = False
        for seed in SEEDS:
            total_seeds_checked += 1
            state = status_dict[seed]
            if "成功" in state:
                success_seeds += 1
            else:
                failed_seeds += 1
                target_failed = True
                # 分类
                err_text = state.replace("失败: ", "")
                label = classify_error(err_text)
                error_counter[label] += 1

        if target_failed:
            failed_targets += 1

        # 写入详细结果
        result_lines.append(f"[{target_id}] (来自 {sub})")
        for seed in SEEDS:
            result_lines.append(f"  seed_{seed}: {status_dict[seed]}")
        result_lines.append("")  # 空行

    # 4. 生成汇总
    summary = []
    summary.append("========== 预测结果检查报告 ==========")
    summary.append(f"基础目录: {BASE_DIR}")
    summary.append(f"数据集划分: {SUBDIRS}")
    summary.append(f"不重复目标总数: {total_unique}")
    if duplicate_flag:
        summary.append(f"发现 {len(duplicates)} 个目标在多个数据集中重复出现:")
        for tid, splits in duplicates.items():
            summary.append(f"  - {tid}: 出现在 {', '.join(splits)}")
    else:
        summary.append("所有目标无重复，train/val/test 划分干净。")
    summary.append("")
    summary.append(f"检查种子数/目标: {len(SEEDS)}")
    summary.append(f"总检查 seed 数: {total_seeds_checked}")
    summary.append(f"成功: {success_seeds}")
    summary.append(f"失败: {failed_seeds}")
    summary.append(f"存在失败的目标数: {failed_targets}")
    summary.append("")
    summary.append("--- 错误分类统计 ---")
    if error_counter:
        for err, count in error_counter.items():
            summary.append(f"  {err}: {count} 次")
    else:
        summary.append("  无一失败。")
    summary.append("")
    summary.append("========== 详细记录 ==========")

    # 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary) + '\n\n')
        f.write('\n'.join(result_lines))

    # 屏幕输出关键信息
    for line in summary:
        print(line)
    print(f"详细结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()