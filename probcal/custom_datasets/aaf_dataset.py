import os
from pathlib import Path
from typing import Callable
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class AAFDataset(Dataset):
    """The All-Age-Faces (AAF) Dataset contains 13'322 face images (mostly Asian)
    distributed across all ages (from 2 to 80), including 7381 females and 5941 males."""

    # DATA_URL = "https://drive.google.com/uc?id=1wa5qOHUZn9O3Zp1efVoTwkK7PWZu8FJS"

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test"],
        limit: int | None = None,
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
        surface_image_path: bool = False,
    ):
        """Create an instance of the AAF dataset.

        Args:
            root_dir (str | Path): Root directory where dataset files should be stored.
            limit (int | None, optional): Max number of images to download/use for this dataset. Defaults to None.
            transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version. e.g, `transforms.RandomCrop`. Defaults to None.
            target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
            surface_image_path (bool, optional): Whether/not to return the image path along with the image and age in __getitem__.
        """

        self.root_dir = Path(root_dir)
        self.split = split
        self.limit = limit
        self.image_dir = self.root_dir / "images"
        self.annotations_csv_path = self.root_dir / "annotations.csv"
        self.surface_image_path = surface_image_path

        for dir in self.root_dir, self.image_dir:
            if not dir.exists():
                os.makedirs(dir)

        if not self._already_downloaded():
            raise Exception(
                "Dataset is not present in the specified location. Contact the authors for access."
            )

        self.transform = transform
        self.target_transform = target_transform
        self.instances = self._get_instances_df()

    def _already_downloaded(self) -> bool:
        return (
            self.annotations_csv_path.exists()
            and self.image_dir.exists()
            and any(self.image_dir.iterdir())
        )

    def _get_instances_df(self) -> pd.DataFrame:
        annotations = str(self.annotations_csv_path)
        df = pd.read_csv(annotations)
        df = df[df["split"] == self.split]
        return df

    def __getitem__(self, idx: int) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        try:
            row = self.instances.iloc[idx]
        except Exception:
            row = self.instances.iloc[idx.item()]
        image_path = self.image_dir / self.split / row["image_path"]
        image = Image.open(image_path)
        age = row["age"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            age = self.target_transform(age)
        if self.surface_image_path:
            return image, (image_path, age)
        else:
            return image, age

    def __len__(self):
        return len(self.instances)
