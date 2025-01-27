from pathlib import Path
from typing import Callable
from typing import Literal

import torch
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage


class RotatedMNIST(Dataset):
    """A dataset made up of rotated MNIST images."""

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[float], float] | None = None,
    ):
        """Initialize a RotatedMNIST dataset.

        Args:
            root_dir (str | Path): Directory where dataset files are stored.
            split (str): The dataset split to load (either train, val, or test).
            transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
        """
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / split
        self.n = len(list(self.data_dir.iterdir()))
        self.transform = transform
        self.target_transform = target_transform
        self.to_pil = ToPILImage()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image, angle = torch.load(self.data_dir / f"{idx}.pt", weights_only=True)
        image = self.to_pil(image.repeat(3, 1, 1))  # Simulate RGB

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            image = self.target_transform(angle)

        return image, angle
