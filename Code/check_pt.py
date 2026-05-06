import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def check_dataset_health(data_root: str):
    print(f"🔍 正在扫描数据目录: {data_root}")

    # 获取所有的 .pt 文件
    pt_files = list(Path(data_root).rglob("*.pt"))
    total_files = len(pt_files)
    print(f"📂 共找到 {total_files} 个 .pt 文件。开始逐一读取测试...\n")

    if total_files == 0:
        print("❌ 未找到任何数据，请检查路径。")
        return

    ratios = []
    corrupted_files = []

    # 逐一读取测试，使用 try-except 防止个别坏文件导致整个程序崩溃
    for f in tqdm(pt_files, desc="读取测试"):
        try:
            # 尝试加载数据字典
            data = torch.load(f)

            # 校验关键键值是否存在
            if 'pos' not in data or 'pos_pred' not in data:
                raise ValueError("缺失关键坐标(pos)字段")

            # 记录对齐率
            ratio = data.get('alignment_ratio', 0.0)
            ratios.append(ratio)

        except Exception as e:
            corrupted_files.append((str(f), str(e)))

    # ---------------- 打印健康报告 ----------------
    print("\n" + "=" * 50)
    print("📊 数据集健康体检报告")
    print("=" * 50)
    print(f"✅ 成功读取: {len(ratios)} / {total_files} 个文件")

    if corrupted_files:
        print(f"❌ 损坏/读取失败: {len(corrupted_files)} 个文件")
        print("前 5 个损坏的文件及原因:")
        for path, err in corrupted_files[:5]:
            print(f"  - {path.split('/')[-1]}: {err}")
    else:
        print("🎉 恭喜！没有任何文件损坏，100% 可正常读取！")

    # ---------------- 统计与可视化 ----------------
    if ratios:
        ratios = np.array(ratios)
        mean_ratio = np.mean(ratios)
        median_ratio = np.median(ratios)
        min_ratio = np.min(ratios)

        print("\n📈 序列对齐率 (Alignment Ratio) 统计:")
        print(f"  - 平均对齐率: {mean_ratio:.2%}")
        print(f"  - 中位数: {median_ratio:.2%}")
        print(f"  - 最低对齐率: {min_ratio:.2%}")
        print(f"  - 对齐率 > 90% 的占比: {(ratios > 0.9).mean():.2%}")

        # 绘制直方图并保存 (因为在服务器上直接 plt.show() 可能会报错)
        plt.figure(figsize=(10, 6))

        # 排除 0，以免有些异常文件拉低了正常可视化的比例尺
        valid_ratios = ratios[ratios > 0]

        plt.hist(valid_ratios, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of RNA Atom Alignment Ratios', fontsize=16)
        plt.xlabel('Alignment Ratio (Aligned Atoms / Total GT Atoms)', fontsize=14)
        plt.ylabel('Frequency (Number of .pt files)', fontsize=14)
        plt.grid(axis='y', alpha=0.75)

        # 标出平均线
        plt.axvline(mean_ratio, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ratio:.2f}')
        plt.legend()

        # 保存图片到数据同级目录
        save_path = os.path.join(data_root, 'alignment_ratio_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n🖼️ 对齐率分布直方图已生成并保存至: {save_path}")
        print("你可以把这张图片下载到本地查看整体的数据质量分布。")


if __name__ == "__main__":
    # 配置你的数据集根目录
    DATA_ROOT = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"

    check_dataset_health(DATA_ROOT)