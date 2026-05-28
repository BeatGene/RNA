from pathlib import Path
from typing import Dict

import lightning.pytorch as pl
from torch_geometric.loader import DataLoader

# ToDo
from .dataset import EuclideanDataset


class BaseDataModule(pl.LightningDataModule):
    """Datamodule to do all data stuff."""

    def __init__(
        self,
        data_dir: Path | None = None,
        dataloader_args: Dict = {},
        sample_files: str | list[str] | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataloader_args = dataloader_args

        # sample_files 可以是一个文件路径(每行一个.pt文件) 或 直接的列表
        if isinstance(sample_files, str):
            with open(sample_files, "r") as f:
                self.sample_files = [line.strip() for line in f if line.strip()]
        else:
            self.sample_files = sample_files

    def __repr__(self) -> str:
        return f"RNARefinementDataModule(data_dir={self.data_dir})"

    def setup(self, stage: str = None):
        self.train_dataset = EuclideanDataset(
            self.data_dir,  split="train", sample_files=self.sample_files
        )
        self.val_dataset = EuclideanDataset(
            self.data_dir,  split="val", sample_files=self.sample_files
        )

        self.test_dataset = EuclideanDataset(
            self.data_dir,split="test",
        )

    def train_dataloader(self):
        """Creates train dataloader"""
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_args)

    def val_dataloader(self):
        """Creates val dataloader"""
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_args)
