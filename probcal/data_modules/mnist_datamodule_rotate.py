from pathlib import Path

from matplotlib import pyplot as plt

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomRotation  # Add this import



class MNISTDataModuleRotate(L.LightningDataModule):
    def __init__(
        self, root_dir: str | Path, batch_size: int, num_workers: int, persistent_workers: bool
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.test_rotation = 0

    def setup(self, stage: str):
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.mnist_test = MNIST(self.root_dir, train=False, download=True, transform=transform)
        self.mnist_predict = MNIST(self.root_dir, train=False, download=True, transform=transform)
        mnist_full = MNIST(self.root_dir, train=True, download=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(1998)
        )

    def set_test_rotation(self, degrees: float) -> None:
        """Set the rotation angle for test data."""
        self.test_rotation = degrees

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_rotation != 0:
            test_transform = Compose([
                ToTensor(),
                RandomRotation(degrees=(self.test_rotation, self.test_rotation)),  # Fixed angle rotation
                Normalize((0.1307,), (0.3081,)),
            ])
            self.mnist_test.transform = test_transform

        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
    
#write me a test script for this datamodule so we can see how we are rotating images, lets print out a few images as well

if __name__ == '__main__':
    print("Creating datamodule")
    datamodule = MNISTDataModuleRotate(root_dir="data", batch_size=4, num_workers=8, persistent_workers=True)
    print("Setting up datamodule")
    datamodule.setup("test")
    print("Setting test rotation")
    datamodule.set_test_rotation(degrees=180)
    print("Getting train loader")
    test_loader = datamodule.test_dataloader()
    print("Printing out a few images")
    i = 0
    for batch in test_loader:
        images, _ = batch
        #save the image as a png using matplotlib
        plt.imshow(images[0].squeeze().numpy(), cmap="gray")
        plt.savefig(f"test_image_{i}.png")
        i += 1
        if i > 3:
            break