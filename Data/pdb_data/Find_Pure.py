import os
import pandas as pd

# 路径设置
BASE_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data"
PURE_RNA_DIR = os.path.join(BASE_DIR, "01_Pure_RNA")
INPUT_CSV = os.path.join(BASE_DIR, "all_rna_structures.csv")
OUTPUT_CSV = "/remote-home/jinxianwang/tinghaoxia/RNA/Code/pure_rna_metadata.csv"


def filter_csv():
    # 1. 获取 01_Pure_RNA 目录下所有的 PDB ID
    # 假设文件名是 '9yd3.cif'，我们需要提取 '9yd3'
    if not os.path.exists(PURE_RNA_DIR):
        print(f"错误: 找不到目录 {PURE_RNA_DIR}")
        return

    # 提取文件名（去除后缀，并转为大写以便匹配）
    pure_rna_ids = {
        os.path.splitext(f)[0].upper()
        for f in os.listdir(PURE_RNA_DIR)
        if f.endswith('.cif')
    }

    print(f"在 01_Pure_RNA 文件夹中找到了 {len(pure_rna_ids)} 个唯一的 PDB ID。")

    # 2. 读取原始全量 CSV 文件
    if not os.path.exists(INPUT_CSV):
        print(f"错误: 找不到 CSV 文件 {INPUT_CSV}")
        return

    df_all = pd.read_csv(INPUT_CSV)

    # 3. 执行筛选
    # 确保 CSV 中的 PDB_ID 也转为大写进行比对，防止大小写不一致
    df_pure = df_all[df_all['PDB_ID'].str.upper().isin(pure_rna_ids)]

    # 4. 保存结果
    df_pure.to_csv(OUTPUT_CSV, index=False)

    print(f"筛选完成！")
    print(f"原始记录数: {len(df_all)}")
    print(f"纯 RNA 记录数: {len(df_pure)}")
    print(f"结果已保存至: {OUTPUT_CSV}")

    # 打印前几行示例
    if not df_pure.empty:
        print("\n前 5 行预览:")
        print(df_pure.head())


if __name__ == "__main__":
    filter_csv()