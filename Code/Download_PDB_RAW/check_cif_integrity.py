import os
import sys
import warnings
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ============ 配置 ============
PDB_DATA_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data"
SUBDIRS = [
    "01_Pure_RNA",
    "02_RNA_Protein_Complex",
    "03_Ribosome_Apo",
    "04_Ribosome_Bound_RNA",
    "05_Others_or_Failed",
]
# =============================


def check_cif_integrity():
    """对前 4 个文件夹列出解析失败的，对第 5 个文件夹列出解析成功的。"""
    results = {}

    for subdir in SUBDIRS:
        subdir_path = os.path.join(PDB_DATA_DIR, subdir)
        if not os.path.exists(subdir_path):
            print(f"跳过不存在的目录: {subdir_path}")
            continue

        cif_files = sorted([f for f in os.listdir(subdir_path) if f.endswith(".cif")])
        ok_list, fail_list = [], []

        print(f"\n正在检查 [{subdir}]，共 {len(cif_files)} 个文件...")
        for f in tqdm(cif_files, desc=subdir, unit="file"):
            filepath = os.path.join(subdir_path, f)
            try:
                d = MMCIF2Dict(filepath)
                _ = d["_entry.id"]
                ok_list.append(f)
            except Exception:
                fail_list.append(f)
                tqdm.write(f"  ❌ 解析失败: {subdir}/{f}")

        results[subdir] = {"ok": ok_list, "fail": fail_list}
        print(f"  → 成功 {len(ok_list)}，失败 {len(fail_list)}")

    # 前 4 个文件夹：列出失败的
    print("\n" + "=" * 60)
    print("前 4 个文件夹中【解析失败】的文件：")
    print("=" * 60)
    found_any = False
    for subdir in SUBDIRS[:4]:
        fail = results.get(subdir, {}).get("fail", [])
        if fail:
            found_any = True
            for f in fail:
                print(f"  {subdir}/{f}")
    if not found_any:
        print("  无失败文件 ✓")

    # 第 5 个文件夹：列出成功的
    print("\n" + "=" * 60)
    print("05_Others_or_Failed 中【解析成功】的文件：")
    print("=" * 60)
    subdir5 = SUBDIRS[4]
    ok = results.get(subdir5, {}).get("ok", [])
    if ok:
        for f in ok:
            print(f"  {subdir5}/{f}")
        print(f"  共 {len(ok)} 个 — 这些文件解析正常，可能被误判到 others 中")
    else:
        print("  无成功文件")


if __name__ == "__main__":
    check_cif_integrity()
