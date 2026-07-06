"""
排查代表链 FASTA 到 .clstr 之间的 test PDB 缺失。

用法：
  python check_clstr_gap.py
"""

import os

FASTA_REP = "./split_analysis/representative_chains.fasta"
CLSTR_PATH = "./split_analysis/rep_clusters_0.90.clstr"


def parse_header(header):
    parts = header.split("_")
    split = parts[-1]
    pdb = "_".join(parts[:-2])
    return pdb, split


def main():
    # 1. 代表链 FASTA 中 test PDB
    fasta_test = set()
    with open(FASTA_REP) as f:
        for line in f:
            if line.startswith(">"):
                pdb, split = parse_header(line[1:].strip())
                if split == "test":
                    fasta_test.add(pdb)
    print(f"代表链 FASTA 中 test PDB: {len(fasta_test)}")

    # 2. .clstr 中 test PDB
    clstr_test = set()
    if not os.path.exists(CLSTR_PATH):
        print(f"错误: {CLSTR_PATH} 不存在")
        return

    with open(CLSTR_PATH) as f:
        for line in f:
            if not line.startswith(">Cluster") and ">" in line:
                parts = line.split(">")[1].split("...")[0].strip().split("_")
                split = parts[-1]
                pdb = "_".join(parts[:-2])
                if split == "test":
                    clstr_test.add(pdb)
    print(f".clstr 中 test PDB: {len(clstr_test)}")

    # 3. 差异
    missing = fasta_test - clstr_test
    extra = clstr_test - fasta_test
    print(f"\n代表链 FASTA 有但 .clstr 没有: {len(missing)} 个")
    if missing:
        print("  示例:", sorted(missing)[:30])

    print(f"\n.clstr 有但代表链 FASTA 没有: {len(extra)} 个")
    if extra:
        print("  示例:", sorted(extra)[:10])

    # 4. 检查缺失 PDB 的链长分布
    if missing:
        lengths = {}
        with open(FASTA_REP) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                pdb, split = parse_header(line[1:].strip())
                if split == "test" and pdb in missing:
                    seq = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    lengths[pdb] = len(seq)
        print(f"\n缺失 PDB 的链长分布:")
        for pdb in sorted(missing)[:10]:
            print(f"  {pdb}: {lengths.get(pdb, '?')} nt")


if __name__ == "__main__":
    main()
