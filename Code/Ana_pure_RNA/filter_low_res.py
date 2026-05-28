import os
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def _get_first(cif_dict, key, default=None):
    val = cif_dict.get(key, default)
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    return val


def _clean_resolution(raw):
    """清洗分辨率值，返回 float 或 None"""
    if raw is None:
        return None
    if raw in ['?', '.', 'None', 'N/A']:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def get_resolution(filepath):
    """
    从 CIF 文件中提取分辨率。
    依次尝试 X-ray 字段、电镜字段。
    """
    cif_dict = MMCIF2Dict(filepath)

    raw = _get_first(cif_dict, '_refine.ls_d_res_high',
                     _get_first(cif_dict, '_reflns.d_resolution_high', None))

    if raw is None:
        # 冷冻电镜: _em_3d_reconstruction.resolution
        raw = _get_first(cif_dict, '_em_3d_reconstruction.resolution', None)

    return _clean_resolution(raw)


def filter_low_resolution(cif_dir, threshold=5.0, output_txt='low_resolution_files.txt'):
    """
    扫描目录下所有 .cif 文件，将 Resolution > threshold 的文件名写入 txt。

    Parameters
    ----------
    cif_dir : str
        CIF 文件所在目录路径
    threshold : float
        分辨率阈值（单位 Å），大于此值的被认为粗糙
    output_txt : str
        输出文件名
    """
    cif_files = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
    total = len(cif_files)
    print(f"扫描目录: {cif_dir}")
    print(f"找到 {total} 个 CIF 文件，阈值 = {threshold} Å\n")

    low_res = []   # 分辨率 > threshold
    no_res = []    # 没有分辨率值（比如 NMR）
    stats = []

    for i, filename in enumerate(cif_files):
        filepath = os.path.join(cif_dir, filename)
        pdb_id = filename.split('.')[0][:4].upper()
        res = get_resolution(filepath)

        if res is None:
            no_res.append(filename)
            stats.append((pdb_id, filename, 'N/A', '无分辨率'))
        elif res > threshold:
            low_res.append(filename)
            stats.append((pdb_id, filename, res, '粗糙'))
        # res <= threshold: 合格，不记录

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i + 1}/{total}...")

    # 写入结果
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"# Resolution > {threshold} Å 的 CIF 文件\n")
        f.write(f"# 共 {len(low_res)} 个\n")
        f.write(f"# {'='*60}\n")
        for pdb_id, filename, res, _ in stats:
            if _ == '粗糙':
                f.write(f"{filename}\t{pdb_id}\t{res} Å\n")

    # 终端汇总
    print(f"\n{'='*50}")
    print(f"扫描完成: {total} 个文件")
    print(f"  分辨率 > {threshold} Å (粗糙) : {len(low_res)}")
    print(f"  分辨率 <= {threshold} Å (合格) : {total - len(low_res) - len(no_res)}")
    print(f"  无分辨率值 (NMR等)           : {len(no_res)}")
    print(f"\n结果已写入: {output_txt}")

    # 如果有粗糙的，打印前 20 个
    if low_res:
        print(f"\n粗糙文件列表 (前 20):")
        for pdb_id, filename, res, _ in stats:
            if _ == '粗糙':
                print(f"  {filename:40s} {pdb_id}  {res} Å")


if __name__ == '__main__':
    CIF_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/pdb_data/01_Pure_RNA"
    OUTPUT = "low_resolution_files.txt"
    THRESHOLD = 5.0

    if os.path.exists(CIF_DIR):
        filter_low_resolution(CIF_DIR, threshold=THRESHOLD, output_txt=OUTPUT)
    else:
        print(f"[!] 目录不存在: {CIF_DIR}")
