from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from probcal.data_modules.probcal_datamodule import ProbcalDataModule


class TabularDataModule(ProbcalDataModule):
    def __init__(
        self,
        dataset_path: str | Path,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
    ):
        self.dataset_path = Path(dataset_path)
        super().__init__(
            root_dir=self.dataset_path.parent,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def prepare_data(self):
        pass

    def setup(self, stage):
        data: dict[str, np.ndarray] = np.load(self.dataset_path)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
            X_val = X_val.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)

        self.train = TensorDataset(
            torch.Tensor(X_train),
            torch.Tensor(y_train).unsqueeze(1),
        )
        self.val = TensorDataset(
            torch.Tensor(X_val),
            torch.Tensor(y_val).unsqueeze(1),
        )
        self.test = TensorDataset(
            torch.Tensor(X_test),
            torch.Tensor(y_test).unsqueeze(1),
        )
