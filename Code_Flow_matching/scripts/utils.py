from loguru import logger as log
from typing import List, Optional
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from datetime import datetime
import os
from lightning.pytorch.utilities import rank_zero_only
# ToDo
from etflow.models.model import BaseFlow
import yaml

def instantiate_logger(
    logger_type: str,
    logger_args: dict,
    task_name: str,
    debug_mode: bool = False,
    no_logger: bool = False,
) -> Logger:
    if debug_mode or no_logger:
        return None

    if logger_type == "WandbLogger":
        if "name" in logger_args:
            del logger_args["name"]  # name is set by task_name
        logger = WandbLogger(**logger_args, name=task_name)
    else:
        raise NotImplementedError

    return logger


def instantiate_model(
    model_type: str, model_args: dict,
) -> LightningModule:
    if model_type == "BaseFlow":
        log.info(f"Loading BaseFlow with args: {model_args}")
        return BaseFlow(**model_args)

    raise NotImplementedError

def instantiate_callbacks(callbacks: list) -> List[Callback]:
    final_callbacks = []
    for callback_dict in callbacks:
        if callback_dict["callback"] == "ModelCheckpoint":
            final_callbacks.append(ModelCheckpoint(**callback_dict["callback_args"]))
        elif callback_dict["callback"] == "EarlyStopping":
            final_callbacks.append(EarlyStopping(**callback_dict["callback_args"]))
        elif callback_dict["callback"] == "LearningRateMonitor":
            final_callbacks.append(
                LearningRateMonitor(**callback_dict["callback_args"])
            )
        else:
            raise NotImplementedError

    return final_callbacks

def instantiate_trainer(
    trainer_type: str,
    trainer_args: dict,
    logger: Logger,
    callbacks: List[Callback],
    debug: bool,
) -> Trainer:
    if debug:
        trainer_args["fast_dev_run"] = 2
        trainer_args["devices"] = 1  # check on single GPU
        trainer_args["strategy"] = "auto"  # auto select strategy

    if trainer_type == "Trainer":
        trainer = Trainer(**trainer_args, logger=logger, callbacks=callbacks)
    else:
        raise NotImplementedError

    return trainer

def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_log_dir():
    """
    Check if LOG_DIR is env variable is set
    else use logs/ directory
    """
    log_dir = os.environ.get("LOG_DIR")
    if log_dir is None:
        log_dir = "logs/"

    return log_dir

def setup_log_dir(task_name, project_root=None):
    """Sets log directory for a given task
    and then moves to that directory
    """
    log_dir = get_log_dir()
    if not os.path.isabs(log_dir) and project_root is not None:
        log_dir = os.path.join(project_root, log_dir)
    # use time to create unique log directory
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, task_name, "runs", f"{run_name}")
    log.info(f"Log directory: {log_dir}")

    # create log directory
    os.makedirs(log_dir, exist_ok=True)
    os.chdir(log_dir)

@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]
    hparams["model_args"] = cfg["model_args"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["datamodule_args"] = cfg["datamodule_args"]
    hparams["trainer"] = cfg["trainer"]
    hparams["trainer_args"] = cfg["trainer_args"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)