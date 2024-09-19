from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class EVADataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[PILImage], PILImage] | None = None,
        target_transform: Callable[[int], int] | None = None,
        surface_image_path: bool = False,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.surface_image_path = surface_image_path

        self.root_dir = Path(root_dir)

        # read the label data in from the data folder
        self.labels_dir = self.root_dir.joinpath("data")
        self.votes_filtered_df = pd.read_csv(
            self.labels_dir.joinpath("votes_filtered.csv"), sep="="
        )

        # find the average of all the votes to create the label for the image
        self.labels_df = (
            self.votes_filtered_df.groupby("image_id")["score"]
            .agg(["mean", "count", "std"])
            .reset_index()
        )
        self.labels_df.columns = ["image_id", "avg_score", "vote_count", "score_std"]

        # the file name for each image is just {image_id}.jpg
        self.labels_df["file_name"] = self.labels_df["image_id"].astype(str) + ".jpg"
        self.image_dir = self.root_dir.joinpath("images", "EVA_together")

    def __getitem__(self, idx: int) -> tuple[PILImage, int] | tuple[PILImage, tuple[str, int]]:
        row = self.labels_df.iloc[idx]
        image_path = self.image_dir.joinpath(row["file_name"])
        image = Image.open(image_path)
        score = row["avg_score"]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            score = self.target_transform(score)
        if self.surface_image_path:
            return image, (image_path, score)
        else:
            return image, score

    def __len__(self):
        return len(self.labels_df)

    def _print_stats(self):
        print("\nStatistics of average scores:")
        print(self.labels_df["avg_score"].describe())

    def _create_dist_graph(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.labels_df, x="avg_score", kde=True)
        plt.title("Distribution of Average Scores per Image")
        plt.xlabel("Average Score")
        plt.ylabel("Count")
        plt.show()


eva = EVADataset(root_dir="../../data/eva-dataset", surface_image_path=True)
print(eva.labels_df.head())
print(len(eva))
eva._print_stats()
print(eva[0])
