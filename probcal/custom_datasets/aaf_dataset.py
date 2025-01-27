import csv
import os
from pathlib import Path
from shutil import move
from shutil import rmtree
from shutil import unpack_archive
from typing import Callable
from typing import Literal


import gdown
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class AAFDataset(Dataset):
    """The All-Age-Faces (AAF) Dataset contains 13'322 face images (mostly Asian)
    distributed across all ages (from 2 to 80), including 7381 females and 5941 males."""

    DATA_URL = "https://drive.google.com/uc?id=1wa5qOHUZn9O3Zp1efVoTwkK7PWZu8FJS"

    def __init__(
        self,
        root_dir: str | Path,
        # split: Literal["train", "val", "test"],
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
        # self.split = split
        self.limit = limit
        self.image_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"
        self.annotations_csv_path = self.annotations_dir / "annotations.csv"
        self.surface_image_path = surface_image_path

        for dir in self.root_dir, self.image_dir, self.annotations_dir:
            if not dir.exists():
                os.makedirs(dir)

        if not self._already_downloaded():
            self._download()

        self.transform = transform
        self.target_transform = target_transform
        self.instances = self._get_instances_df()
        self.annotations_df = pd.read_csv(self.annotations_csv_path)

    def _already_downloaded(self) -> bool:
        return (
            self.annotations_dir.exists()
            and any(self.annotations_dir.iterdir())
            and self.image_dir.exists()
            and any(self.image_dir.iterdir())
        )

    def _download(self):

        # Download raw folder
        print("Downloading zipped file...")
        zip_file_name = "All-Age-Faces Dataset"
        output_path = str(self.root_dir) + "/" + zip_file_name + ".zip"
        gdown.download(self.DATA_URL, output_path, quiet=False, fuzzy=True)
        unpack_archive(output_path, self.root_dir)

        # Setup annotations files
        input_txt_file1 = str(self.root_dir / zip_file_name / "image sets" / "train.txt")
        input_txt_file2 = str(self.root_dir / zip_file_name / "image sets" / "val.txt")
        output_csv_file = str(self.annotations_csv_path)

        # Convert from txt to csv + transformations
        with open(input_txt_file1, "r") as txt_file1, open(
            input_txt_file2, "r"
        ) as txt_file2, open(output_csv_file, "w") as csv_file:

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["image_path", "age", "gender"])

            for line in txt_file1:
                filename, gender = line.strip().split()
                age = filename[-6:-4]
                csv_writer.writerow([filename, age, gender])

            for line in txt_file2:
                filename, gender = line.strip().split()
                age = filename[-6:-4]
                csv_writer.writerow([filename, age, gender])

        # Setup images folder
        src_folder = str(self.root_dir / zip_file_name / "original images")

        for filename in os.listdir(src_folder):
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(self.image_dir, filename)
            move(src_file, dest_file)

        # Clean up
        os.remove(output_path)
        rmtree(str(self.root_dir / zip_file_name))

    def _get_instances_df(self) -> pd.DataFrame:
        annotations = str(self.annotations_csv_path)
        return pd.read_csv(annotations)

    def __getitem__(self, idx: int) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        row = self.instances.iloc[idx]
        image_path = str(self.image_dir / row["image_path"])
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


aaf_data = AAFDataset(root_dir='data/aaf')#, split='train')
print("Dataset length:")
print(f"{len(aaf_data.annotations_df)} images")

num_instances = len(aaf_data)
generator = np.random.default_rng(seed=117)
shuffled_indices = generator.permutation(np.arange(num_instances))
num_train = int(0.7 * num_instances)
num_val = int(0.1 * num_instances)
train_indices = shuffled_indices[:num_train]
val_indices = shuffled_indices[num_train : num_train + num_val]
test_indices = shuffled_indices[num_train + num_val :]

train_dir = Path("data/aaf/images/train")
val_dir = Path("data/aaf/images/val")
test_dir = Path("data/aaf/images/test")
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# add split column to the labels_df
aaf_data.annotations_df["split"] = ""

# move the files and update the labels_df
for idx, indices in enumerate([train_indices, val_indices, test_indices]):
    for i in indices:
        row = aaf_data.annotations_df.iloc[i]
        image_path = aaf_data.image_dir.joinpath(row["image_path"])
        if idx == 0:
            image_path.rename(train_dir.joinpath(row["image_path"]))
            aaf_data.annotations_df.loc[i, "split"] = "train"
        elif idx == 1:
            image_path.rename(val_dir.joinpath(row["image_path"]))
            aaf_data.annotations_df.loc[i, "split"] = "val"
        else:
            image_path.rename(test_dir.joinpath(row["image_path"]))
            aaf_data.annotations_df.loc[i, "split"] = "test"

# check the number of images in each folder
print(len(list(train_dir.glob("*.jpg"))))
print(len(list(val_dir.glob("*.jpg"))))
print(len(list(test_dir.glob("*.jpg"))))


# write the labels_df to a csv file
aaf_data.annotations_df.to_csv("data/aaf/labels.csv", index=False)
