from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import GaussianBlur
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from probcal.custom_datasets.image_dataset_wrapper import LabelNoiseImageDatasetWrapper
from probcal.custom_datasets.image_dataset_wrapper import MixupImageDatasetWrapper
from probcal.data_modules.probcal_datamodule import ProbcalDataModule
from probcal.transforms import GaussianNoiseTransform
from probcal.transforms import MixUpTransform


class OodBlurDataModule(ProbcalDataModule):
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
        self.test = self._get_test_set(
            self.root_dir, inference_transforms, self.surface_image_path
        )

    def _get_test_set(
        self, root_dir: str | Path, transform: Compose, surface_image_path: bool
    ) -> Dataset:
        raise NotImplementedError("Must be implemented by subclass.")


class OodMixupDataModule(ProbcalDataModule):
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
        test_dataset = self._get_test_set(
            self.root_dir, inference_transforms, self.surface_image_path
        )
        self.test = MixupImageDatasetWrapper(
            base_dataset=test_dataset,
            transforms=inference_transforms,
            mixup_transform=mixup_transform,
        )

    def _get_test_set(
        self, root_dir: str | Path, transform: Compose, surface_image_path: bool
    ) -> Dataset:
        raise NotImplementedError("Must be implemented by subclass.")


class OodLabelNoiseDataModule(ProbcalDataModule):
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
        test_dataset = self._get_test_set(
            self.root_dir, inference_transforms, self.surface_image_path
        )
        noise_transform = GaussianNoiseTransform(**kwargs["perturb"])
        self.test = LabelNoiseImageDatasetWrapper(
            base_dataset=test_dataset,
            transforms=inference_transforms,
            noise_transform=noise_transform,
        )

    def _get_test_set(
        self, root_dir: str | Path, transform: Compose, surface_image_path: bool
    ) -> Dataset:
        raise NotImplementedError("Must be implemented by subclass.")
