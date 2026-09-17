import argparse
import os.path as osp
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from lightning.pytorch import seed_everything
from loguru import logger as log

from utils import (
    instantiate_callbacks,
    instantiate_logger,
    instantiate_model,
    instantiate_trainer,
    log_hyperparameters,
    read_yaml,
    setup_log_dir,
)

from etflow.data.datamodule import BaseDataModule

torch.set_float32_matmul_precision("high")


def validate_config(config: dict) -> None:
    """Fail before creating a WandB run when the training setup is invalid."""
    for key in ("task_name", "datamodule_args", "model", "model_args",
                "callbacks", "trainer", "trainer_args"):
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    data_dir = Path(config["datamodule_args"]["data_dir"]).expanduser()
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing PT split directory: {split_dir}")
    config["datamodule_args"]["data_dir"] = str(data_dir.resolve())

    loader_args = config["datamodule_args"].get("dataloader_args", {})
    batch_size = int(loader_args.get("batch_size", 1))
    num_workers = int(loader_args.get("num_workers", 0))
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if loader_args.get("persistent_workers", False) and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")

    pretrained_ckpt = config.get("pretrained_ckpt")
    resume_ckpt = config.get("ckpt_path")
    if pretrained_ckpt and resume_ckpt:
        raise ValueError("Set only one of pretrained_ckpt and ckpt_path")
    for label, checkpoint in (
        ("pretrained_ckpt", pretrained_ckpt),
        ("ckpt_path", resume_ckpt),
    ):
        if checkpoint:
            checkpoint_path = Path(checkpoint).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {checkpoint}")
            config[label] = str(checkpoint_path.resolve())

    trainer_args = config["trainer_args"]
    if trainer_args.get("accelerator") == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("trainer accelerator is gpu but CUDA is unavailable")
        devices = 1 if config.get("debug", False) else trainer_args.get("devices", 1)
        if isinstance(devices, int) and torch.cuda.device_count() < devices:
            raise RuntimeError(
                f"Requested {devices} GPUs but only "
                f"{torch.cuda.device_count()} are visible"
            )

    logger_args = config.get("logger_args") or {}
    if logger_args.get("save_dir"):
        wandb_dir = Path(logger_args["save_dir"]).expanduser()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        logger_args["save_dir"] = str(wandb_dir.resolve())


def run(config: dict) -> None:
    validate_config(config)

    # check if debug mode
    debug = config.get("debug", False)

    # seed everything for reproducibility
    seed_everything(config.get("seed", 42), workers=True)

    # task name for logger, if not provided use default
    task_name = config.get("task_name", None)
    if debug:
        task_name = "debug-run"
    assert task_name is not None, "Task name not provided"

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
    setup_log_dir(task_name)

    # instantiate datamodule

    datamodule = BaseDataModule(**config["datamodule_args"])

    # instantiate model
    # ToDo 注意node_attr_dim、edge_attr_dim参数的修改
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
    callback_specs = config["callbacks"]
    if logger is None:
        # LearningRateMonitor requires a logger.  This keeps --debug and
        # --no_logger usable for local pipeline checks.
        callback_specs = [
            spec for spec in callback_specs
            if spec.get("callback") != "LearningRateMonitor"
        ]
    callbacks = instantiate_callbacks(callback_specs)

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
    if not osp.isfile(args.config):
        parser.error(f"Config file not found: {args.config}")
    config = read_yaml(args.config)

    # update config with debug mode
    config["debug"] = args.debug
    config["no_logger"] = args.no_logger
    run(config)
