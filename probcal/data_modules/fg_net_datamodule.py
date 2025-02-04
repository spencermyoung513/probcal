from pathlib import Path

from torchvision.transforms import AutoAugment
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from probcal.custom_datasets import FGNetDataset
from probcal.data_modules.ood_datamodule import OodBlurDataModule
from probcal.data_modules.ood_datamodule import OodLabelNoiseDataModule
from probcal.data_modules.ood_datamodule import OodMixupDataModule
from probcal.data_modules.probcal_datamodule import ProbcalDataModule


class FGNetDataModule(ProbcalDataModule):

    IMG_SIZE = 224
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        ignore_grayscale: bool = True,
        surface_image_path: bool = False,
    ):
        super().__init__(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        self.surface_image_path = surface_image_path
        self.ignore_grayscale = ignore_grayscale

    def prepare_data(self) -> None:
        FGNetDataset(self.root_dir, split="train")

    def setup(self, stage: str):
        resize = Resize((self.IMG_SIZE, self.IMG_SIZE))
        augment = AutoAugment()
        normalize = Normalize(mean=self.IMAGE_NET_MEAN, std=self.IMAGE_NET_STD)
        to_tensor = ToTensor()
        train_transforms = Compose([resize, augment, to_tensor, normalize])
        inference_transforms = Compose([resize, to_tensor, normalize])

        self.train = FGNetDataset(
            root_dir=self.root_dir,
            split="train",
            transform=train_transforms,
            ignore_grayscale=self.ignore_grayscale,
            surface_image_path=self.surface_image_path,
        )
        self.val = FGNetDataset(
            root_dir=self.root_dir,
            split="val",
            transform=inference_transforms,
            ignore_grayscale=self.ignore_grayscale,
            surface_image_path=self.surface_image_path,
        )
        self.test = FGNetDataset(
            root_dir=self.root_dir,
            split="test",
            transform=inference_transforms,
            ignore_grayscale=self.ignore_grayscale,
            surface_image_path=self.surface_image_path,
        )


class OodBlurFGNetDataModule(OodBlurDataModule, FGNetDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        ignore_grayscale: bool = True,
        surface_image_path: bool = False,
    ):
        super().__init__(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            surface_image_path=surface_image_path,
            ignore_grayscale=ignore_grayscale,
        )

    def _get_test_set(self, root_dir: str | Path, transform: Compose, surface_image_path: bool):
        return FGNetDataset(
            root_dir=root_dir,
            split="test",
            transform=transform,
            surface_image_path=surface_image_path,
        )


class OodMixupFGNetDataModule(OodMixupDataModule, FGNetDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        ignore_grayscale: bool = True,
        surface_image_path: bool = False,
    ):
        super().__init__(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            ignore_grayscale=ignore_grayscale,
            surface_image_path=surface_image_path,
        )

    def _get_test_set(self, root_dir: str | Path, transform: Compose, surface_image_path: bool):
        return FGNetDataset(root_dir=root_dir, split="test", surface_image_path=surface_image_path)


class OodLabelNoiseFGNetDataModule(OodLabelNoiseDataModule, FGNetDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        ignore_grayscale: bool = True,
        surface_image_path: bool = False,
    ):
        super().__init__(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            ignore_grayscale=ignore_grayscale,
            surface_image_path=surface_image_path,
        )

    def _get_test_set(self, root_dir: str | Path, transform: Compose, surface_image_path: bool):
        return FGNetDataset(
            root_dir=root_dir,
            split="test",
            transform=transform,
            surface_image_path=surface_image_path,
        )
