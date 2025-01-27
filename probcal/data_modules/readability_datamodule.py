import os
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers import DistilBertTokenizer

from probcal.custom_datasets import ReadabilityDataset
from probcal.data_modules.probcal_datamodule import ProbcalDataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")


class ReadabilityDataModule(ProbcalDataModule):
    def prepare_data(self) -> None:
        # Force check if reviews are already downloaded.
        ReadabilityDataset(self.root_dir, split="train")

    def setup(self, stage):
        self.train = ReadabilityDataset(self.root_dir, split="train")
        self.val = ReadabilityDataset(self.root_dir, split="val")
        self.test = ReadabilityDataset(self.root_dir, split="test")

    def train_dataloader(self) -> DataLoader:
        if self.train is None:
            raise ValueError("The `train` attribute has not been set. Did you call `setup` yet?")
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val is None:
            raise ValueError("The `val` attribute has not been set. Did you call `setup` yet?")
        return self.get_dataloader(
            split="val",
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test is None:
            raise ValueError("The `test` attribute has not been set. Did you call `setup` yet?")
        return self.get_dataloader(
            split="test",
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    @staticmethod
    def collate_fn(batch: Iterable[tuple[str, int]]) -> tuple[BatchEncoding, torch.LongTensor]:
        all_text = []
        all_targets = []
        for x in batch:
            all_text.append(x[0])
            all_targets.append(x[1])

        input_tokens = tokenizer(
            all_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        targets_tensor = torch.tensor(all_targets)

        return input_tokens, targets_tensor
