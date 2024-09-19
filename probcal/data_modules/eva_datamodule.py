from pathlib import Path

import lightning as L

from probcal.custom_datasets import EVADataset


class EVADataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        surface_image_path: bool = False,
    ):
        print("hello from the EVA datamodule!")
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.surface_image_path = surface_image_path
        self.prepare_data()

    def prepare_data(self) -> None:
        # prepare/extract the images from the zip files
        print("hello from prepare data")
        EVADataset(self.root_dir)


eva_dm = EVADataModule("../../data/eva-dataset", 0, 0, False)
