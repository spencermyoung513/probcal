from pathlib import Path

from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

from probcal.custom_datasets import RotatedMNIST
from probcal.data_modules.probcal_datamodule import ProbcalDataModule


class RotatedMNISTDataModule(ProbcalDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
    ):
        super().__init__(
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def prepare_data(self) -> None:
        # Force check if dataset is already present.
        RotatedMNIST(self.root_dir, split="train")

    def setup(self, stage: str):
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train = RotatedMNIST(self.root_dir, split="train", transform=transform)
        self.val = RotatedMNIST(self.root_dir, split="val", transform=transform)
        self.test = RotatedMNIST(self.root_dir, split="test", transform=transform)
