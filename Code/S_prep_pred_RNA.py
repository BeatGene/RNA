import os
import glob
import subprocess
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区域 =================

# 1. 路径配置
INPUT_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Simple_json/01_Pure_RNA"
BASE_OUTPUT_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"

# 数据库路径
DB_SEQRES = "/remote-home/share/protenix_database/af2_data/pdb_seqres"
DB_RFAM = "/remote-home/share/protenix_database/af2_data/rfam/Rfam.fasta"
DB_RNACENTRAL = "/remote-home/share/protenix_database/af2_data/rnacentral/rnacentral_active.fasta"
DB_NTRNA = "/remote-home/share/protenix_database/af2_data/ntrna/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"

# 2. 硬件与并发配置
# 根据你的 nvidia-smi，8张卡都可以用。但注意每张卡似乎已经被占用了 9GB 显存。
# 如果预测任务显存大于 14GB，可能会 OOM。你可以先设置少量 GPU 测试，例如 [4]
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_WORKERS = len(AVAILABLE_GPUS)  # 并发数等于可用显卡数

# 3. 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("protenix_batch_run.log"),
        logging.StreamHandler()
    ]
)

# ================= 核心逻辑 =================

# 创建 GPU 队列
gpu_queue = queue.Queue()
for gpu in AVAILABLE_GPUS:
    gpu_queue.put(gpu)


def run_command(cmd, env=None):
    """执行 shell 命令并捕获输出"""
    try:
        # 将命令列表转换为字符串，方便日志打印
        cmd_str = " ".join([str(x) for x in cmd])
        logging.info(f"正在执行: {cmd_str}")

        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout


def process_single_rna(json_path):
    """处理单个 RNA json 文件的完整流程 (Prep -> Pred)"""
    filename = os.path.basename(json_path)
    prefix = os.path.splitext(filename)[0]  # 例如: 9zca

    # 申请 GPU
    gpu_id = gpu_queue.get()

    try:
        logging.info(f"[{prefix}] 开始处理，分配到 GPU: {gpu_id}")

        # --- 步骤 1: 运行 protenix prep ---
        prep_out_dir = os.path.join(BASE_OUTPUT_DIR, f"prep_output_{prefix}")
        os.makedirs(prep_out_dir, exist_ok=True)

        prep_cmd = [
            "protenix", "prep",
            "-i", json_path,
            "-o", prep_out_dir,
            "-m", "protenix",
            "--seqres_database_path", DB_SEQRES,
            "--rfam_database_path", DB_RFAM,
            "--rna_central_database_path", DB_RNACENTRAL,
            "--ntrna_database_path", DB_NTRNA
        ]

        success, prep_log = run_command(prep_cmd)
        if not success:
            logging.error(f"[{prefix}] Prep 阶段失败!\n{prep_log}")
            return False

        logging.info(f"[{prefix}] Prep 完成.")

        # --- 步骤 2: 运行 protenix pred ---
        # 假设 prep 生成的更新文件在 prep_out_dir 下，名为 {prefix}-final-updated.json
        # 如果 protenix 默认生成在当前运行目录，请修改此路径
        updated_json_path = os.path.join(INPUT_DIR, f"{prefix}-final-updated.json")

        if not os.path.exists(updated_json_path):
            logging.error(f"[{prefix}] 找不到 Prep 生成的文件: {updated_json_path}")
            return False

        pred_out_dir = os.path.join(BASE_OUTPUT_DIR, f"pred_output_{prefix}")
        os.makedirs(pred_out_dir, exist_ok=True)

        pred_cmd = [
            "protenix", "pred",
            "-i", updated_json_path,
            "-o", pred_out_dir,
            "-n", "protenix_base_default_v1.0.0",
            "--use_msa", "True",
            "--use_rna_msa", "True",
            "--use_template", "True",
            "--dtype", "bf16",
            "--sample", "5",
            "--step", "200",
            "--cycle", "10",
            "--enable_cache", "True"
        ]

        # 设置环境变量，限制只能看到分配给它的那一块 GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        success, pred_log = run_command(pred_cmd, env)
        if not success:
            logging.error(f"[{prefix}] Pred 阶段失败!\n{pred_log}")
            return False

        logging.info(f"[{prefix}] Pred 成功完成! 结果保存在: {pred_out_dir}")
        return True

    except Exception as e:
        logging.error(f"[{prefix}] 发生未知错误: {str(e)}")
        return False

    finally:
        # 无论成功失败，释放 GPU 回队列
        gpu_queue.put(gpu_id)
        logging.info(f"[{prefix}] 释放 GPU: {gpu_id}")


def main():
    # 查找所有以 2 开头的 json 文件
    search_pattern = os.path.join(INPUT_DIR, "2*.json")
    json_files = sorted(glob.glob(search_pattern))

    if not json_files:
        logging.error(f"在 {INPUT_DIR} 中没有找到以 2 开头的 json 文件！")
        return

    logging.info(f"找到 {len(json_files)} 个匹配的文件。开始使用 {MAX_WORKERS} 个并发任务...")

    # 使用线程池并发执行
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_rna, f) for f in json_files]

        # 等待完成
        for future in futures:
            future.result()

    total_time = time.time() - start_time
    logging.info(f"所有任务处理完毕！总耗时: {total_time / 3600:.2f} 小时.")


if __name__ == "__main__":
    main()