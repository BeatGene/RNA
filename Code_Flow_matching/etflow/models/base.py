from typing import Dict, Optional, Union

import torch
from lightning.pytorch import LightningModule
from loguru import logger as log
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from etflow.schedulers import CosineAnnealingWarmupRestarts


class BaseModel(LightningModule):
    def __init__(
        self,
        # optimizer
        optimizer_type: str = "Adam",
        lr: float = 1e-3,
        beta1: float = 0.95,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        ams_grad: bool = False,
        grad_norm_max_val: float = 100.0,
        # lr scheduler args
        lr_scheduler_type: Optional[str] = "plateau",
        factor: float = 0.6,
        patience: int = 10,
        first_cycle_steps: int = 1000,
        cycle_mult: float = 1.0,
        max_lr: float = 0.0001,
        min_lr: float = 1.0e-08,
        warmup_steps: int = 10000,
        gamma: float = 0.75,
        last_epoch: int = -1,
        lr_scheduler_monitor: str = "val/loss",
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_frequency: int = 1,
    ):
        super().__init__()

        # optimizer
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.opt_betas = (
            beta1,
            beta2,
        )
        self.weight_decay = weight_decay
        self.ams_grad = ams_grad
        self.grad_norm_max_val = grad_norm_max_val

        # lr scheduler
        self.lr_scheduler_type = lr_scheduler_type
        self.factor = factor
        self.patience = patience
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.lrs_gamma = gamma
        self.last_epoch = last_epoch
        self.lr_scheduler_monitor = lr_scheduler_monitor
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_frequency = lr_scheduler_frequency

    def generic_step(self, batch, batch_idx: int, mode: str):
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int):
        return self.generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self.generic_step(batch, batch_idx, "val")

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        if self.optimizer_type == "Adam":
            log.info(f"Using Adam optimizer with lr={self.lr}")
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                betas=self.opt_betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "AdamW":
            log.info(f"Using AdamW optimizer with lr={self.lr}")
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                betas=self.opt_betas,
                weight_decay=self.weight_decay,
                amsgrad=self.ams_grad,
            )
        else:
            log.info(f"Using SGD optimizer with lr={self.lr}")
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
            )

        if self.lr_scheduler_type is not None:
            if self.lr_scheduler_type == "plateau":
                log.info(
                    f"Using ReduceLROnPlateau with factor={self.factor}, patience={self.patience}"
                )
                lr_scheduler_config = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=self.optimizer,
                        factor=self.factor,
                        patience=self.patience,
                    ),
                    "monitor": self.lr_scheduler_monitor,
                    "interval": self.lr_scheduler_interval,
                    "frequency": self.lr_scheduler_frequency,
                }
            elif self.lr_scheduler_type == "CosineAnnealingWarmupRestarts":
                log.info(
                    f"Using CosineAnnealingWarmupRestarts with"
                    f"first_cycle_steps={self.first_cycle_steps}, cycle_mult={self.cycle_mult},"
                    f"max_lr={self.max_lr}, min_lr={self.min_lr}, warmup_steps={self.warmup_steps},"
                    f"gamma={self.lrs_gamma}, last_epoch={self.last_epoch}"
                )
                lr_scheduler_config = {
                    "scheduler": CosineAnnealingWarmupRestarts(
                        optimizer=self.optimizer,
                        first_cycle_steps=self.first_cycle_steps,
                        cycle_mult=self.cycle_mult,
                        max_lr=self.max_lr,
                        min_lr=self.min_lr,
                        warmup_steps=self.warmup_steps,
                        gamma=self.lrs_gamma,
                        last_epoch=self.last_epoch,
                    ),
                    "interval": self.lr_scheduler_interval,
                    "frequency": self.lr_scheduler_frequency,
                }
            else:
                raise ValueError(f"Unknown lr_scheduler_type {self.lr_scheduler_type}")

            return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

        log.info("Using no lr scheduler")
        return {"optimizer": self.optimizer}

    def log_helper(self, key: str, value: torch.Tensor, batch_size: int):
        is_train_metric = key.startswith("train/")
        self.log(
            key,
            value,
            # Training curves are indexed by optimizer progress, while
            # validation metrics are reduced once per validation epoch.
            on_step=is_train_metric,
            on_epoch=not is_train_metric,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            # Avoid one distributed all-reduce per training metric and batch.
            # Validation must be synchronized to obtain a global checkpoint
            # metric over all DDP ranks.
            sync_dist=not is_train_metric,
        )

    def configure_gradient_clipping(
            self,
            optimizer,
            gradient_clip_val,
            gradient_clip_algorithm,
    ):
        """Apply fixed global-norm clipping configured by the Trainer.

        Lightning delegates clipping to this hook when a model overrides it.
        The previous implementation ignored ``gradient_clip_val`` from the
        Trainer, so the YAML value of 1.0 was not actually the clipping
        threshold.  Keep ``grad_norm_max_val`` only as an additional safety
        ceiling for backward-compatible configs.
        """
        max_grad_norm = min(float(gradient_clip_val), self.grad_norm_max_val)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=max_grad_norm,
            norm_type=2.0,
            error_if_nonfinite=True,
        )
        clip_fraction = torch.clamp(
            grad_norm.new_tensor(max_grad_norm) / grad_norm.clamp_min(1.0e-12),
            max=1.0,
        )
        self.log(
            "train/grad_norm_unclipped",
            grad_norm,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            sync_dist=False,
        )
        self.log(
            "train/grad_clip_fraction",
            clip_fraction,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            sync_dist=False,
        )
