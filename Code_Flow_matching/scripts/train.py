import argparse
import os
import os.path as osp
from pathlib import Path

import torch
from lightning.pytorch import seed_everything
from loguru import logger as log
# ToDo
from utils import (
    instantiate_callbacks,
    instantiate_logger,
    instantiate_model,
    instantiate_trainer,
    log_hyperparameters,
    read_yaml,
    setup_log_dir,
)
# ToDo
from etflow.data.datamodule import BaseDataModule

torch.set_float32_matmul_precision("high")

# NCCL timeout settings to prevent multi-GPU deadlocks
os.environ.setdefault("NCCL_TIMEOUT", "1800")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_DISTRIBUTED_TIMEOUT", "1800")


def run(config: dict) -> None:
    # check if debug mode
    debug = config.get("debug", False)

    # seed everything for reproducibility
    seed_everything(config.get("seed", 42))

    # 保存项目根目录 (setup_log_dir 会 os.chdir 到日志目录)
    project_root = os.getcwd()

    # task name for logger, if not provided use default
    task_name = config.get("task_name", None)
    if debug:
        task_name = "debug-run"
    assert task_name is not None, "Task name not provided"

    # 提前解析 sample_files 的相对路径，防止 os.chdir 后找不到
    dm_args = config.get("datamodule_args", {})
    if "sample_files" in dm_args and isinstance(dm_args["sample_files"], str):
        sf_path = Path(dm_args["sample_files"])
        if not sf_path.is_absolute():
            dm_args["sample_files"] = str((Path(project_root) / sf_path).resolve())

    # instantiate logger (skip if debug mode)
    logger = None
    if config.get("logger") is not None:
        logger = instantiate_logger(
            config.get("logger"),
            config.get("logger_args"),
            task_name=task_name,
            debug_mode=debug,
            no_logger=config.get("no_logger", False),
        )

    # setup log directory (pass project_root to ensure absolute path)
    setup_log_dir(task_name, project_root=project_root)

    # instantiate datamodule
    datamodule = BaseDataModule(**dm_args)

    # instantiate model
    model = instantiate_model(config["model"], config["model_args"])
    pretrained_ckpt = config.get("pretrained_ckpt", None)
    if pretrained_ckpt is not None:
        assert osp.exists(
            pretrained_ckpt
        ), f"Pretrained checkpoint {pretrained_ckpt} not found!"
        state_dict = torch.load(pretrained_ckpt, map_location=model.device)[
            "state_dict"
        ]
        model.load_state_dict(state_dict)
        log.info(f"Loaded pretrained model from checkpoint: {pretrained_ckpt}")

    # instantiate callbacks
    callbacks = instantiate_callbacks(config["callbacks"])

    # instantiate trainer
    trainer = instantiate_trainer(
        config["trainer"],
        config["trainer_args"],
        logger=logger,
        callbacks=callbacks,
        debug=debug,
    )

    # log config
    log_hyperparameters({"cfg": config, "model": model, "trainer": trainer})

    # start training
    resume_ckpt_path = config.get("ckpt_path", None)
    if resume_ckpt_path is not None:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    # read config path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--no_logger", "-n", action="store_true")
    args = parser.parse_args()

    # read config
    osp.exists(args.config), f"Config file {args.config} not found"
    config = read_yaml(args.config)

    # update config with debug mode
    config["debug"] = args.debug
    config["no_logger"] = args.no_logger
    run(config)
