import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 你的 JSON 文件夹路径
folder_path = '/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Simple_json/01_Pure_RNA'

# 用于存储解析后的结果，使用字典可以按 name 去重
results_dict = {}

# 存储所有单独链条的长度，用于最后画图
all_chain_lengths_for_plot = []

print("开始扫描并解析 JSON 文件...")

for filename in os.listdir(folder_path):
    if not filename.endswith('.json'):
        continue

    file_path = os.path.join(folder_path, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # 处理外层是列表的情况，如 [{...}]
            if isinstance(data, list):
                if len(data) == 0:
                    continue
                data = data[0]

            name = data.get('name', 'unknown')
            sequences = data.get('sequences', [])

            chain_types = []
            chain_lengths = []

            # 遍历所有的序列
            for seq_item in sequences:
                seq_type = ""
                seq_info = {}

                # 判断链条种类
                if 'rnaSequence' in seq_item:
                    seq_type = 'RNA'
                    seq_info = seq_item['rnaSequence']
                elif 'proteinSequence' in seq_item:
                    seq_type = 'Protein'
                    seq_info = seq_item['proteinSequence']
                elif 'dnaSequence' in seq_item:
                    seq_type = 'DNA'
                    seq_info = seq_item['dnaSequence']
                else:
                    continue

                seq_str = seq_info.get('sequence', '')
                count = seq_info.get('count', 1)
                length = len(seq_str)

                # 如果 count 大于 1，说明存在多条完全相同的链
                for _ in range(count):
                    chain_types.append(seq_type)
                    chain_lengths.append(str(length))

            # 处理重复问题：如果 name 已经存在，我们优先保留文件名中带有 'updated' 的数据
            is_updated_file = 'updated' in filename.lower()

            if name not in results_dict or is_updated_file:
                results_dict[name] = {
                    'name': name,
                    '链条种类': ' '.join(chain_types),
                    '链条长度': ' '.join(chain_lengths),
                    '_raw_lengths': [int(l) for l in chain_lengths]  # 暂存用于画图
                }

        except Exception as e:
            print(f"读取文件出错 {filename}: {e}")

# 整理数据到 DataFrame
df_data = []
for name, info in results_dict.items():
    df_data.append({
        'name': info['name'],
        '链条种类': info['链条种类'],
        '链条长度': info['链条长度']
    })
    # 把当前 name 下的所有链条长度加入画图大列表
    all_chain_lengths_for_plot.extend(info['_raw_lengths'])

# 1. 导出为 Excel 文件
df = pd.DataFrame(df_data)
excel_output_path = 'rna_chain_info.xlsx'
df.to_excel(excel_output_path, index=False)
print(f"✅ Excel 汇总文件已生成: {excel_output_path} (共 {len(df)} 条不重复数据)")

# 2. 绘制整体长度分布直方图
plt.figure(figsize=(10, 6))
# 绘制直方图，bins=30 表示将长度分为30个区间
plt.hist(all_chain_lengths_for_plot, bins=30, color='#4A90E2', edgecolor='black', alpha=0.8)

plt.title('Distribution of RNA Chain Lengths', fontsize=16, pad=15)
plt.xlabel('Chain Length (Nucleotides)', fontsize=14)
plt.ylabel('Frequency (Number of Chains)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图片
plot_output_path = 'chain_length_distribution.png'
plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
print(f"✅ 长度分布图已生成: {plot_output_path}")