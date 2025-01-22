from typing import Literal

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset


class BootstrapMixin:
    """Provides functionality for a `LightningDataModule` to resample its val/test splits (e.g., for estimating the variability of a metric)."""

    def __init__(self):
        self.bootstrap_indices = {"val": None, "test": None}

    def set_bootstrap_indices(
        self, split: Literal["val", "test"], indices: torch.LongTensor | None
    ):
        """
        Set bootstrap indices for a specific split.

        Args:
            split (Literal["val", "test"]): The split to resample.
            indices (torch.LongTensor | None): Indices to use for bootstrapping, or None to reset.
        """
        if split not in self.bootstrap_indices:
            raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'.")
        self.bootstrap_indices[split] = indices

    def get_dataloader(
        self, split: Literal["val", "test"], dataset: Dataset, **kwargs
    ) -> DataLoader:
        """Return a DataLoader for the specified split, reindexed with bootstrap indices if set.

        Args:
            split (Literal["val", "test"]): The split to load (e.g., 'val' or 'test').
            dataset (Dataset): The dataset corresponding to the split.
            **kwargs: Additional arguments for the DataLoader (e.g., batch_size, num_workers).

        Returns:
            DataLoader: A DataLoader for the bootstrap-resampled or regular dataset split.
        """
        if split not in self.bootstrap_indices:
            raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'.")

        indices = self.bootstrap_indices[split]
        if indices is not None:
            dataset = Subset(dataset, indices)
        return DataLoader(dataset, **kwargs)
