from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import AutoAugment
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from probcal.custom_datasets import AAFDataset
from probcal.custom_datasets import ImageDatasetWrapper


class AAFDataModule(L.LightningDataModule):

    IMG_SIZE = 224

    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.surface_image_path = surface_image_path

    def setup(self, stage):
        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        augment = AutoAugment()
        normalize = Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )  # Check the mean and variance of our images
        to_tensor = ToTensor()
        train_transforms = Compose([resize, augment, to_tensor, normalize])
        inference_transforms = Compose([resize, to_tensor, normalize])

        full_dataset = AAFDataset(
            self.root_dir,
            surface_image_path=self.surface_image_path,
        )
        num_instances = len(full_dataset)
        generator = np.random.default_rng(seed=1998)
        shuffled_indices = generator.permutation(np.arange(num_instances))
        num_train = int(0.7 * num_instances)
        num_val = int(0.1 * num_instances)
        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train : num_train + num_val]
        test_indices = shuffled_indices[num_train + num_val :]

        self.train = ImageDatasetWrapper(
            base_dataset=Subset(full_dataset, train_indices),
            transforms=train_transforms,
        )
        self.val = ImageDatasetWrapper(
            base_dataset=Subset(full_dataset, val_indices),
            transforms=inference_transforms,
        )
        self.test = ImageDatasetWrapper(
            base_dataset=Subset(full_dataset, test_indices),
            transforms=inference_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
