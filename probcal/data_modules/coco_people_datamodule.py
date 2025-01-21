from pathlib import Path

import lightning as L
from deep_uncertainty.custom_datasets import COCOPeopleDataset
from torch.utils.data import DataLoader
from torchvision.transforms import AutoAugment
from torchvision.transforms import Compose
from torchvision.transforms import GaussianBlur
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from probcal.custom_datasets.image_dataset_wrapper import LabelNoiseImageDatasetWrapper
from probcal.custom_datasets.image_dataset_wrapper import MixupImageDatasetWrapper
from probcal.transforms import GaussianNoiseTransform
from probcal.transforms import MixUpTransform


class COCOPeopleDataModule(L.LightningDataModule):

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

    def prepare_data(self) -> None:
        # Force check if images are already downloaded.
        COCOPeopleDataset(self.root_dir, split="train")

    def setup(self, stage):
        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        augment = AutoAugment()
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        train_transforms = Compose([resize, augment, to_tensor, normalize])
        inference_transforms = Compose([resize, to_tensor, normalize])

        self.train = COCOPeopleDataset(
            root_dir=self.root_dir,
            split="train",
            transform=train_transforms,
            surface_image_path=self.surface_image_path,
        )
        self.val = COCOPeopleDataset(
            root_dir=self.root_dir,
            split="val",
            transform=inference_transforms,
            surface_image_path=self.surface_image_path,
        )
        self.test = COCOPeopleDataset(
            root_dir=self.root_dir,
            split="test",
            transform=inference_transforms,
            surface_image_path=self.surface_image_path,
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


class OodBlurCocoPeopleDataModule(COCOPeopleDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(root_dir, batch_size, num_workers, persistent_workers, surface_image_path)

    def setup(self, stage, *args, **kwargs):
        if stage != "test":
            raise ValueError(f"Invalid stage: {stage}. Only 'test' is supported for OOD class")

        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        ood_transform = GaussianBlur(
            kernel_size=(5, 9), sigma=(kwargs["perturb"], kwargs["perturb"])
        )
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        inference_transforms = Compose([resize, ood_transform, to_tensor, normalize])
        self.test = COCOPeopleDataset(
            self.root_dir,
            split="test",
            transform=inference_transforms,
            surface_image_path=self.surface_image_path,
        )


class OodMixupCocoPeopleDataModule(COCOPeopleDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(root_dir, batch_size, num_workers, persistent_workers, surface_image_path)

    def setup(self, stage, *args, **kwargs):
        if stage != "test":
            raise ValueError(f"Invalid stage: {stage}. Only 'test' is supported for OOD class")

        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        inference_transforms = Compose([resize, to_tensor, normalize])
        mixup_transform = MixUpTransform(kwargs["perturb"])
        test_dataset = COCOPeopleDataset(
            self.root_dir,
            split="test",
            surface_image_path=self.surface_image_path,
        )
        self.test = MixupImageDatasetWrapper(
            base_dataset=test_dataset,
            transforms=inference_transforms,
            mixup_transform=mixup_transform,
        )


class OodLabelNoiseCocoPeopleDataModule(COCOPeopleDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        super().__init__(root_dir, batch_size, num_workers, persistent_workers, surface_image_path)

    def setup(self, stage, *args, **kwargs):
        if stage != "test":
            raise ValueError(f"Invalid stage: {stage}. Only 'test' is supported for OOD class")

        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        to_tensor = ToTensor()
        inference_transforms = Compose([resize, to_tensor, normalize])
        test_dataset = COCOPeopleDataset(
            self.root_dir,
            split="test",
            surface_image_path=self.surface_image_path,
        )
        noise_transform = GaussianNoiseTransform(**kwargs["perturb"])
        self.test = LabelNoiseImageDatasetWrapper(
            base_dataset=test_dataset,
            transforms=inference_transforms,
            noise_transform=noise_transform,
        )
