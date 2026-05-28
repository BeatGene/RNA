"""
从数据集中随机挑选 N 个固定的 .pt 样本，输出其路径列表。
用于后续只在这些固定样本上训练/验证。

用法 (在服务器上运行):
  python scripts/select_samples.py \
      --data_dir /remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA \
      --split train \
      --num_samples 10 \
      --seed 42 \
      --output config/sample_10_files.txt
"""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="config/sample_10_files.txt")
    args = parser.parse_args()

    random.seed(args.seed)

    split_dir = Path(args.data_dir) / args.split
    all_pt_files = sorted(split_dir.rglob("*.pt"))

    print(f"Found {len(all_pt_files)} .pt files in {split_dir}")

    selected = random.sample(all_pt_files, min(args.num_samples, len(all_pt_files)))

    print(f"Selected {len(selected)} files:")
    for f in selected:
        print(f"  {f}")

    # 写入文件，每行一个绝对路径
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for f in selected:
            fh.write(f"{f}\n")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
