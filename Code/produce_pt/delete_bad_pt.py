import os
import pandas as pd
from glob import glob

DATA_ROOT = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"

df = pd.read_csv('rmsd_all.csv')
bad = df[df['rmsd'] > 50]

print(f"RMSD > 50 Å 的记录: {len(bad)} 条\n")

deleted = 0
not_found = 0

for _, row in bad.iterrows():
    split_dir = os.path.join(DATA_ROOT, row['split'], row['pdb_id'])
    # 匹配该 pdb_id + seed 的所有 sample .pt 文件
    pattern = os.path.join(split_dir, f"{row['pdb_id']}_s{row['seed']}_*.pt")
    files = glob(pattern)
    if not files:
        not_found += 1
        continue
    for f in files:
        os.remove(f)
        deleted += 1

print(f"已删除: {deleted} 个 .pt 文件")
if not_found:
    print(f"未找到对应文件: {not_found} 条记录")
