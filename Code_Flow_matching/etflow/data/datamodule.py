from pathlib import Path
from typing import Dict, Optional
import math

import lightning.pytorch as pl
import torch
from torch_geometric.loader import DataLoader

from .dataset import EuclideanDataset
from .length_sampling import long_rna_weights


class BaseDataModule(pl.LightningDataModule):
    """Datamodule to do all data stuff."""

    def __init__(
        self,
        data_dir: Path | None = None,
        dataloader_args: Optional[Dict] = None,
        train_length_table: Path | None = None,
        long_rna_oversample_factor: float = 1.0,
        long_rna_threshold_nt: int = 50,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataloader_args = dict(dataloader_args or {})
        self.train_length_table = Path(train_length_table) if train_length_table else None
        self.long_rna_oversample_factor = float(long_rna_oversample_factor)
        self.long_rna_threshold_nt = int(long_rna_threshold_nt)
        if (not math.isfinite(self.long_rna_oversample_factor)
                or self.long_rna_oversample_factor < 1
                or self.long_rna_threshold_nt < 1):
            raise ValueError("long-RNA factor must be finite and >= 1; threshold must be positive")
        if self.long_rna_oversample_factor > 1 and self.train_length_table is None:
            raise ValueError("train_length_table is required for long-RNA oversampling")


    def __repr__(self) -> str:
        return "RNARefinementDataModule"

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            self.train_dataset = EuclideanDataset(self.data_dir, split="train")
            self.val_dataset = EuclideanDataset(self.data_dir, split="val")
        elif stage == "validate":
            self.val_dataset = EuclideanDataset(self.data_dir, split="val")
        if stage in (None, "test"):
            self.test_dataset = EuclideanDataset(self.data_dir, split="test")

    def train_dataloader(self):
        """Creates train dataloader"""
        if self.long_rna_oversample_factor > 1:
            weights, counts = long_rna_weights(
                self.train_dataset.data_files,
                Path(self.data_dir) / "train",
                self.train_length_table,
                threshold_nt=self.long_rna_threshold_nt,
                factor=self.long_rna_oversample_factor,
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
            print(f"Long-RNA train sampling: {counts}; factor={self.long_rna_oversample_factor}")
            return DataLoader(self.train_dataset, sampler=sampler, **self.dataloader_args)
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_args)

    def val_dataloader(self):
        """Creates val dataloader"""
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_args)
