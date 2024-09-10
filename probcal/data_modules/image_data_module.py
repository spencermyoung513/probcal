from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self, dataset_path: str | Path, batch_size: int, num_workers: int, persistent_workers: bool
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers