from typing import Type

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.models.backbones import Backbone



class MultiClassNN(L.LightningModule):
    """A neural network that learns a multinomial distribution (conditioned on the input) over a set of discrete classes."""

    def __init__(
        self,
        classes: list[str],
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: OptimizerType,
        optim_kwargs: dict,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a MultiClassNN.

        Args:
            classes (list[str]): The discrete classes this network will learn a distribution over.
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.backbone = backbone_type(**backbone_kwargs)
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=self.num_classes)

        self.head = nn.Linear(self.backbone.output_dim, self.num_classes)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output class logits tensor, with shape (N, `self.num_classes`).
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        return y_hat

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output class logits tensor, with shape (N, `self.num_classes`).
        """
        self.backbone.eval()
        y_hat = self(x)
        self.backbone.train()
        return y_hat
    
    def training_step(self, batch: tuple[torch.Tensor, torch.LongTensor]) -> torch.Tensor:
        x, y = batch
        y = y.squeeze().long()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_acc.update(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y = y.squeeze().long()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.val_acc.update(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_epoch=True)

        return loss

    def test_step(self, batch: torch.Tensor):
        x, y = batch
        y = y.squeeze().long()
        y_hat = self.predict(x)
        self.test_acc.update(y_hat, y)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch
        y_hat = self.predict(x)
        return y_hat

    def configure_optimizers(self) -> dict:
        if self.optim_type == OptimizerType.ADAM:
            optim_class = torch.optim.Adam
        elif self.optim_type == OptimizerType.ADAM_W:
            optim_class = torch.optim.AdamW
        elif self.optim_type == OptimizerType.SGD:
            optim_class = torch.optim.SGD
        optimizer = optim_class(self.parameters(), **self.optim_kwargs)
        optim_dict = {"optimizer": optimizer}

        if self.lr_scheduler_type is not None:
            if self.lr_scheduler_type == LRSchedulerType.COSINE_ANNEALING:
                lr_scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
            lr_scheduler = lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
            optim_dict["lr_scheduler"] = lr_scheduler

        return optim_dict
