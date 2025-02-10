from typing import Literal

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset


class ActiveLearningMixin:
    """Mixin class to add active learning capabilities to any LightningDataModule."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labeled_indices: torch.LongTensor | None = None
        self._unlabeled_indices: torch.LongTensor | None = None
        self._original_train_dataset: Dataset | None = None

    @property
    def train(self) -> Dataset | None:
        """Either the original train dataset passed to this datamodule, or the current subset used for active learning."""
        if hasattr(self, "_active_train_dataset"):
            return self._active_train_dataset
        return self._original_train_dataset if hasattr(self, "_original_train_dataset") else None

    @train.setter
    def train(self, dataset: Dataset):
        """Override of `train` setter such that we cache the original train dataset and update the current active learning subset."""
        if self._original_train_dataset is None:
            self._original_train_dataset = dataset
        self._active_train_dataset = dataset

    def setup(self, initial_labeled_indices: torch.LongTensor):
        """Initialize active learning by setting up labeled (train) and unlabeled datasets.

        Args:
            initial_labeled_indices (torch.LongTensor): Indices of initially labeled samples.
        """
        if self._original_train_dataset is None:
            raise RuntimeError(
                "No training dataset found. Ensure setup() is called before setup_active_learning()"
            )
        self._original_train_dataset = self.train
        self._labeled_indices = initial_labeled_indices

        all_indices = torch.arange(len(self._original_train_dataset))
        self._unlabeled_indices = torch.where(~torch.isin(all_indices, initial_labeled_indices))[0]
        self._active_train_dataset = Subset(self._original_train_dataset, self._labeled_indices)

    def update_labeled_set(self, new_labeled_indices: torch.LongTensor):
        """Update the set of labeled samples in the train set with newly labeled data.

        Args:
            new_labeled_indices (torch.LongTensor): Indices of newly labeled samples
        """
        self._labeled_indices = torch.cat((self._labeled_indices, new_labeled_indices)).unique()
        self._unlabeled_indices = torch.where(
            ~torch.isin(self._unlabeled_indices, self._labeled_indices)
        )[0]
        self.train = Subset(self._original_train_dataset, self._labeled_indices)

    def unlabeled_dataloader(self, **kwargs) -> DataLoader:
        """Get a DataLoader for the unlabeled dataset.

        Returns:
            DataLoader: A torch DataLoader for unlabeled samples.
        """
        if self._unlabeled_indices is None:
            raise RuntimeError(
                "Active learning has not been set up. Call setup_active_learning first."
            )

        unlabeled_dataset = Subset(self._original_train_dataset, self._unlabeled_indices)
        return DataLoader(unlabeled_dataset, **kwargs)

    @property
    def num_labeled(self) -> int:
        """Return the number of labeled samples."""
        return len(self._labeled_indices) if self._labeled_indices is not None else 0

    @property
    def num_unlabeled(self) -> int:
        """Return the number of unlabeled samples."""
        return len(self._unlabeled_indices) if self._unlabeled_indices is not None else 0

    @property
    def labeled_indices(self) -> torch.LongTensor:
        """Return the current labeled indices."""
        return (
            self._labeled_indices.clone()
            if self._labeled_indices is not None
            else torch.tensor([], dtype=torch.long)
        )

    @property
    def unlabeled_indices(self) -> torch.LongTensor:
        """Return the current unlabeled indices."""
        return (
            self._unlabeled_indices.clone()
            if self._unlabeled_indices is not None
            else torch.tensor([], dtype=torch.long)
        )


class ActiveLearningMixin:
    """Mixin to add active learning capabilities to a LightningDataModule."""

    seed_indices: torch.LongTensor | None = None

    def __init__(self, seed_indices: torch.LongTensor | None = None):
        self.labeled_indices = []

        # If fixed indices are provided, use them
        if seed_indices is not None:
            self.seed_indices = seed_indices

    def setup(self, stage: Literal["train", "val", "test"] | None = None):
        """Initialize labeled/unlabeled split at the start, ensuring consistency across runs."""
        if not hasattr(super(), "stage"):
            pass  # Is this right?
        else:
            super().setup(stage)

        # Ensure dataset has a 'labeled' column
        if "labeled" not in self.dataset.instances.columns:
            self.dataset.instances["labeled"] = False

        # Generate shared initial indices if not set
        if self.seed_indices is None:
            total_samples = len(self.dataset.instances)
            num_labeled = int(self.initial_labeled_ratio * total_samples)
            self.seed_indices = (
                np.random.RandomState(42)
                .choice(total_samples, num_labeled, replace=False)
                .tolist()
            )

        # Use the shared initial indices
        self.labeled_indices = self.seed_indices
        self.dataset.instances.loc[self.labeled_indices, "labeled"] = True

        self._update_labeled_dataset()

    def _update_labeled_dataset(self):
        """Update the dataset split for training."""
        labeled_df = self.dataset.instances[self.dataset.instances["labeled"]]
        self.labeled_indices = labeled_df.index.tolist()
        self.labeled_dataset = Subset(self.dataset, self.labeled_indices)

    def get_unlabeled_indices(self) -> list[int]:
        """Return indices of unlabeled samples."""
        return self.dataset.instances.index[~self.dataset.instances["labeled"]].tolist()

    def acquire_samples(self, acquisition_fn, n_samples: int):
        """Select new samples to label based on the acquisition function."""
        unlabeled_indices = self.get_unlabeled_indices()
        selected = acquisition_fn(unlabeled_indices, self.dataset, n_samples)

        # Mark newly acquired samples as labeled
        self.dataset.instances.loc[selected, "labeled"] = True
        self._update_labeled_dataset()
