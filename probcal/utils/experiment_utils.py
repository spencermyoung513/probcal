import random
from pathlib import Path
from typing import Type

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from probcal.data_modules import MNISTDataModule, TabularDataModule
from probcal.enums import DatasetType
from probcal.enums import HeadType
from probcal.models import GaussianNN
from probcal.models import MultiClassNN
from probcal.models import NegBinomNN
from probcal.models import PoissonNN
from probcal.models.backbones import MLP, MNISTCNN
from probcal.utils.configs import TrainingConfig
from probcal.utils.generic_utils import partialclass


def get_model(config: TrainingConfig, return_initializer: bool = False) -> L.LightningModule:

    initializer: Type[L.LightningModule]

    if config.head_type == HeadType.GAUSSIAN:
        initializer = GaussianNN
    elif config.head_type == HeadType.POISSON:
        initializer = PoissonNN
    elif config.head_type == HeadType.NEGATIVE_BINOMIAL:
        initializer = NegBinomNN
    elif config.head_type == HeadType.MULTI_CLASS:
        initializer = MultiClassNN
    else:
        raise ValueError("Invalid head type specified.")

    backbone_kwargs = {}
    if config.dataset_type == DatasetType.TABULAR:
        backbone_type = MLP
        backbone_kwargs["input_dim"] = config.input_dim
        if "spiral" in str(config.dataset_path):
            initializer = partialclass(MultiClassNN, classes=["0", "1"])
    elif config.dataset_type == DatasetType.MNIST:
        backbone_type = MNISTCNN
        initializer = partialclass(MultiClassNN, classes=[str(x) for x in range(10)])
    backbone_kwargs["output_dim"] = config.hidden_dim

    model = initializer(
        backbone_type=backbone_type,
        backbone_kwargs=backbone_kwargs,
        optim_type=config.optim_type,
        optim_kwargs=config.optim_kwargs,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
    )
    if return_initializer:
        return model, initializer
    else:
        return model


def get_datamodule(
    dataset_type: DatasetType,
    dataset_path: Path | None = None,
    batch_size: int = 4,
) -> L.LightningDataModule:
    if dataset_type == DatasetType.TABULAR:
        if dataset_path is None:
            raise ValueError("Must specify path to tabular dataset.")
        return TabularDataModule(
            dataset_path=dataset_path,
            batch_size=batch_size,
            num_workers=9,
            persistent_workers=True,
        )
    elif dataset_type == DatasetType.MNIST:
        return MNISTDataModule(
            root_dir="data/mnist",
            batch_size=batch_size,
            num_workers=9,
            persistent_workers=True,
        )


def fix_random_seed(random_seed: int | None):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)


def get_chkp_callbacks(head_type: HeadType, chkp_dir: Path, chkp_freq: int) -> list[ModelCheckpoint]:
    temporal_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        every_n_epochs=chkp_freq,
        filename="{epoch}",
        save_top_k=-1,
        save_last=True,
    )
    best_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        monitor="val_loss",
        every_n_epochs=1,
        filename="best_loss",
        save_top_k=1,
        enable_version_counter=False,
    )
    best_metric_checkpoint_callback = ModelCheckpoint(
        dirpath=chkp_dir,
        monitor="val_acc" if head_type == HeadType.MULTI_CLASS else "val_mae",
        every_n_epochs=1,
        filename="best_acc" if head_type == HeadType.MULTI_CLASS else "best_mae",
        save_top_k=1,
        enable_version_counter=False,
    )
    return [
        temporal_checkpoint_callback,
        best_loss_checkpoint_callback,
        best_metric_checkpoint_callback,
    ]
