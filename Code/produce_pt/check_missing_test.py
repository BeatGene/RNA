"""
排查测试集 PDB 在 FASTA 中缺失的原因。

用法：
  python check_missing_test.py
"""

import os

FASTA_PATH = "./split_analysis/all_rna_chains.fasta"
TEST_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA/test"
CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"


def parse_fasta_header(header):
    parts = header.split("_")
    split_label = parts[-1]
    pdb_id = "_".join(parts[:-2])
    return pdb_id, split_label


def main():
    # 1. test 目录下的 PDB
    test_pdbs = set()
    for d in os.listdir(TEST_DIR):
        if os.path.isdir(os.path.join(TEST_DIR, d)):
            test_pdbs.add(d)
    print(f"test 目录下 PDB 数: {len(test_pdbs)}")

    # 2. FASTA 中出现的 test PDB
    fasta_test = set()
    with open(FASTA_PATH) as f:
        for line in f:
            if line.startswith(">"):
                pdb_id, split_label = parse_fasta_header(line[1:].strip())
                if split_label == "test":
                    fasta_test.add(pdb_id)
    print(f"FASTA 中 test PDB 数: {len(fasta_test)}")

    # 3. 差异
    missing = test_pdbs - fasta_test
    extra = fasta_test - test_pdbs
    print(f"\n测试集目录有但 FASTA 没有的: {len(missing)} 个")
    if missing:
        print("示例:", sorted(missing)[:30])

    print(f"\nFASTA 有但测试集目录没有的: {len(extra)} 个")
    if extra:
        print("示例:", sorted(extra)[:30])

    # 4. 缺失原因排查
    if missing:
        print(f"\n{'='*60}")
        print("缺失原因排查")
        print(f"{'='*60}")

        no_cif = []
        cif_exists_but_no_chain = []

        for pdb_id in sorted(missing):
            cif_path = os.path.join(CIF_DIR, f"{pdb_id}.cif")
            if not os.path.exists(cif_path):
                no_cif.append(pdb_id)
            else:
                cif_exists_but_no_chain.append(pdb_id)

        print(f"  CIF 文件缺失: {len(no_cif)} 个")
        if no_cif:
            print(f"    示例: {no_cif[:20]}")

        print(f"  CIF 存在但解析不出链 (可能全是氢原子或链长<5): {len(cif_exists_but_no_chain)} 个")
        if cif_exists_but_no_chain:
            print(f"    示例: {cif_exists_but_no_chain[:20]}")

    # 5. 检查 extra
    if extra:
        print(f"\n{'='*60}")
        print("FASTA 多出的 test PDB 原因排查")
        print(f"{'='*60}")
        for pdb_id in sorted(extra)[:10]:
            in_dir = os.path.isdir(os.path.join(TEST_DIR, pdb_id))
            print(f"  {pdb_id}: test目录存在={in_dir}")


if __name__ == "__main__":
    main()
