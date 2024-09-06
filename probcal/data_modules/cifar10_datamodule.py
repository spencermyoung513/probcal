from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor



class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self, root_dir: str | Path, batch_size: int, num_workers: int, persistent_workers: bool
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
    
    def setup(self, stage: str):
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.4915,), (0.2470,)),
            ]
        )

        self.cifar10_test = CIFAR10(self.root_dir, train=False, download=True, transform=transform)
        self.cifar10_predict = CIFAR10(self.root_dir, train=False, download=True, transform=transform)
        cifar10_full = CIFAR10(self.root_dir, train=True, download=True, transform=transform)
        self.cifar10_train, self.cifar10_val = random_split(
            cifar10_full, [45000, 5000], generator=torch.Generator().manual_seed(1998) 
        )
    

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar10_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )