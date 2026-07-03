import os
from collections import defaultdict

from Bio.PDB import MMCIFParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionException
warnings.simplefilter('ignore', PDBConstructionException)

CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
RNA_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"
SPLITS = ["train", "val", "test"]


def main():
    """从 CIF 文件中提取链序列，区分真正丢失的 PDB 和仅有短链的 PDB"""
    parser = MMCIFParser(QUIET=True)
    pdb_chains = defaultdict(dict)

    # 真正丢失的：所有链都 < 5 残基
    no_valid_chain = set()
    # 解析失败的 PDB
    parse_errors = set()
    # 有至少一条短链，但也有合格链的 PDB（不是真正丢失）
    has_short_chain = set()

    cif_files = [f for f in os.listdir(CIF_DIR) if f.endswith(".cif")]
    print(f"解析 {len(cif_files)} 个 CIF 文件...")

    for cif_name in cif_files:
        pdb_id = cif_name.replace(".cif", "")
        cif_path = os.path.join(CIF_DIR, cif_name)
        try:
            structure = parser.get_structure(pdb_id, cif_path)
            model = structure[0]
            pdb_has_short = False
            pdb_has_valid = False
            for chain in model:
                seq = ""
                for res in chain:
                    if res.id[0] != " ":
                        continue
                    res_name = res.get_resname().strip()
                    seq += res_name[0] if res_name else 'N'
                if len(seq) >= 5:
                    pdb_chains[pdb_id][chain.id] = seq
                    pdb_has_valid = True
                else:
                    pdb_has_short = True

            if not pdb_has_valid:
                # 所有链都 < 5 残基，真正丢失
                no_valid_chain.add(pdb_id)
            elif pdb_has_short:
                # 有短链但也有合格链，不算丢失
                has_short_chain.add(pdb_id)
        except Exception:
            parse_errors.add(pdb_id)

    total_cif = len(cif_files)
    total_chains = sum(len(chains) for chains in pdb_chains.values())
    truly_missing = no_valid_chain | parse_errors

    print(f"总 CIF 文件: {total_cif}")
    print(f"提取了 {len(pdb_chains)} 个 PDB, {total_chains} 条链")
    print(f"\n真正丢失的 PDB ({len(truly_missing)} 个) = 无合格链 ({len(no_valid_chain)}) + 解析失败 ({len(parse_errors)})")
    print(f"有短链但未丢失的 PDB: {len(has_short_chain)} 个")
    print(f"验证: {len(pdb_chains)} + {len(truly_missing)} = {len(pdb_chains) + len(truly_missing)} (应为 {total_cif})")

    if parse_errors:
        print(f"\n--- 解析失败的 PDB ({len(parse_errors)} 个) ---")
        for pdb_id in sorted(parse_errors):
            print(f"  {pdb_id}")

    if no_valid_chain:
        print(f"\n--- 所有链均 < 5 残基的 PDB ({len(no_valid_chain)} 个) ---")
        for pdb_id in sorted(no_valid_chain):
            print(f"  {pdb_id}")

    if has_short_chain:
        print(f"\n--- 有短链但仍有合格链的 PDB ({len(has_short_chain)} 个，未丢失) ---")
        for pdb_id in sorted(has_short_chain):
            print(f"  {pdb_id}")


if __name__ == "__main__":
    main()
