import os
import glob
import subprocess
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor

# ================= 配置区域 =================

# 1. 路径配置 (因为跳过了 prep，直接从输入读取 updated json)
INPUT_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Simple_json/01_Pure_RNA"
BASE_OUTPUT_DIR = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/Json_data/Complex_json/01_Pure_RNA"

# 2. 硬件与并发配置
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_WORKERS = len(AVAILABLE_GPUS)  # 8并发

# 3. 预测参数配置 (师兄的要求: 4个种子，每个50个sample)
SEEDS = [42, 43, 44, 45]  # 设置 4 个不同的随机种子
SAMPLES_PER_SEED = "50"  # 每个种子的采样数

# 4. 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("protenix_pred_only_run.log"),
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


def process_single_pred(task_info):
    """只处理 Pred 阶段：接收 (json文件路径, 随机种子)"""
    json_path, seed = task_info

    # 提取原始前缀，例如从 9zca-final-updated.json 提取出 9zca
    filename = os.path.basename(json_path)
    prefix = filename.replace('-final-updated.json', '')

    # 申请 GPU
    gpu_id = gpu_queue.get()

    try:
        logging.info(f"[{prefix} | Seed {seed}] 开始处理，分配到 GPU: {gpu_id}")

        # 为不同的 seed 创建独立的输出文件夹，避免覆盖
        pred_out_dir = os.path.join(BASE_OUTPUT_DIR, f"pred_output_{prefix}_seed_{seed}")
        os.makedirs(pred_out_dir, exist_ok=True)

        pred_cmd = [
            "protenix", "pred",
            "-i", json_path,
            "-o", pred_out_dir,
            "-n", "protenix_base_default_v1.0.0",
            "--use_msa", "True",
            "--use_rna_msa", "True",
            "--use_template", "True",
            "--dtype", "bf16",
            "--sample", SAMPLES_PER_SEED,  # 修改为 50
            "--step", "200",
            "--cycle", "10",
            "--enable_cache", "True",
            "--seeds", str(seed)  # 根据帮助文档，修改为 --seeds
        ]

        # 设置环境变量，限制只能看到分配给它的那一块 GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        success, pred_log = run_command(pred_cmd, env)
        if not success:
            logging.error(f"[{prefix} | Seed {seed}] Pred 阶段失败!\n{pred_log}")
            return False

        logging.info(f"[{prefix} | Seed {seed}] Pred 成功完成! 结果保存在: {pred_out_dir}")
        return True

    except Exception as e:
        logging.error(f"[{prefix} | Seed {seed}] 发生未知错误: {str(e)}")
        return False

    finally:
        # 无论成功失败，释放 GPU 回队列
        gpu_queue.put(gpu_id)
        logging.info(f"[{prefix} | Seed {seed}] 释放 GPU: {gpu_id}")


def main():
    # 查找所有已经完成了 prep 阶段的 updated json 文件
    search_pattern = os.path.join(INPUT_DIR, "*-final-updated.json")
    json_files = sorted(glob.glob(search_pattern))

    if not json_files:
        logging.error(f"在 {INPUT_DIR} 中没有找到任何 -final-updated.json 文件！请确认路径。")
        return

    # 生成所有子任务：每个文件都要跑 4 个 seed
    tasks = []
    for f in json_files:
        for seed in SEEDS:
            tasks.append((f, seed))

    logging.info(
        f"找到 {len(json_files)} 个更新后的文件，共需执行 {len(tasks)} 个预测任务(4个seed/文件)。开始使用 {MAX_WORKERS} 个并发...")

    # 使用线程池并发执行
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_pred, task) for task in tasks]

        # 等待完成
        for future in futures:
            future.result()

    total_time = time.time() - start_time
    logging.info(f"所有预测任务处理完毕！总耗时: {total_time / 3600:.2f} 小时.")


if __name__ == "__main__":
    main()