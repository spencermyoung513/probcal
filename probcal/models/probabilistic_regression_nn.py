from typing import Callable
from typing import Optional
from typing import Type

import lightning as L
import torch
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics.wrappers import BootStrapper

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.models.backbones import Backbone


class ProbabilisticRegressionNN(L.LightningModule):
    """Base class for probabilistic regression neural networks. Should not actually be used for prediction (needs to define `training_step` and whatnot).

    Attributes:
        backbone (Backbone): The backbone to use for feature extraction (before applying the regression head).
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
    """

    def __init__(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: Optional[OptimizerType] = None,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_type: Optional[LRSchedulerType] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        """Instantiate a regression NN.

        Args:
            loss_fn (Callable): The loss function to use for training this NN.
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(ProbabilisticRegressionNN, self).__init__()

        self.backbone = backbone_type(**backbone_kwargs)
        self.optim_type = optim_type
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_fn = loss_fn
        self.train_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.val_mae = MeanAbsoluteError()
        self.test_rmse = BootStrapper(MeanSquaredError(squared=False))
        self.test_mae = BootStrapper(MeanAbsoluteError())

    def configure_optimizers(self) -> dict:
        if self.optim_type is None:
            raise ValueError("Must specify an optimizer type.")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Batched output tensor, with shape (N, D_out)
        """
        return self._forward_impl(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        This method will often differ from `forward` in cases where
        the output used for training is in log (or some other modified)
        space for numerical convenience.

        Args:
            x (torch.Tensor): Batched input tensor, with shape (N, ...).

        Returns:
            torch.Tensor: Batched output tensor, with shape (N, D_out)
        """
        return self._predict_impl(x)

    def sample(
        self, y_hat: torch.Tensor, training: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        """Sample from this network's predictive distributions for a batch of data (as specified by y_hat).

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.
            num_samples (int, optional): Number of samples to take from each predictive distribution. Defaults to 1.

        Returns:
            torch.Tensor: Batched sample tensor, with shape (N, num_samples).
        """
        return self._sample_impl(y_hat, training, num_samples)

    def rsample(
        self,
        y_hat: torch.Tensor,
        training: bool = False,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample (using a differentiable relaxation) from this network's predictive distributions for a batch of data (as specified by y_hat).

        Args:
            y_hat (torch.Tensor): Output tensor from a probabilistic regression network, with shape (N, ...).
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.
            num_samples (int, optional): Number of samples to take from each predictive distribution. Defaults to 1.
            **kwargs: Any additional keyword arguments to pass to the sampling routine, such as `temperature=0.01`.

        Returns:
            torch.Tensor: Batched sample tensor, with shape (N, num_samples).
        """
        return self._rsample_impl(y_hat, training, num_samples, **kwargs)

    def predictive_dist(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.Distribution:
        """Transform the network's outputs into the implied predictive distribution.

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.

        Returns:
            torch.distributions.Distribution: The predictive distribution.
        """
        return self._predictive_dist_impl(y_hat, training)

    def point_prediction(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        """Transform the network's output into a single point prediction.

        This method will vary depending on the type of regression head.
        For example, a gaussian regressor will return the `mean` portion of its output as its point prediction.

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example.

        Returns:
            torch.Tensor: Point predictions for the true regression target, with shape (N, 1).
        """
        return self._point_prediction_impl(y_hat, training)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        with torch.no_grad():
            point_predictions = self.point_prediction(y_hat, training=True).flatten()
            self.train_rmse.update(point_predictions, y.flatten().float())
            self.train_mae.update(point_predictions, y.flatten().float())
            self.log("train_rmse", self.train_rmse, on_epoch=True)
            self.log("train_mae", self.train_mae, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.view(-1, 1).float())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Since we used the model's forward method, we specify training=True to get the proper transforms.
        point_predictions = self.point_prediction(y_hat, training=True).flatten()
        self.val_rmse.update(point_predictions, y.flatten().float())
        self.val_mae.update(point_predictions, y.flatten().float())
        self.log("val_rmse", self.val_rmse, on_epoch=True)
        self.log("val_mae", self.val_mae, on_epoch=True)

        return loss

    def test_step(self, batch: torch.Tensor):
        x, y = batch
        y_hat = self.predict(x)
        point_predictions = self.point_prediction(y_hat, training=False).flatten()
        self.test_rmse.update(point_predictions, y.flatten().float())
        self.test_mae.update(point_predictions, y.flatten().float())
        self._update_addl_test_metrics(x, y_hat, y.view(-1, 1).float())

    def on_test_epoch_end(self):
        test_rmse_result = self.test_rmse.compute()
        test_mae_result = self.test_mae.compute()
        self.log("test_rmse_mean", test_rmse_result["mean"])
        self.log("test_rmse_std", test_rmse_result["std"])
        self.log("test_mae_mean", test_mae_result["mean"])
        self.log("test_mae_std", test_mae_result["std"])
        self._log_addl_test_metrics()

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch
        y_hat = self.predict(x)
        return y_hat

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _sample_impl(
        self, y_hat: torch.Tensor, training: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _rsample_impl(
        self, y_hat: torch.Tensor, training: bool = False, num_samples: int = 1, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _predictive_dist_impl(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.Distribution:
        raise NotImplementedError("Should be implemented by subclass.")

    def _point_prediction_impl(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        raise NotImplementedError("Should be implemented by subclass.")

    def _log_addl_test_metrics(self):
        """Log any test metrics beyond RMSE/MAE (tracked by default)."""
        raise NotImplementedError("Should be implemented by subclass.")

    def _update_addl_test_metrics(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        """Update additional test metric states (beyond default rmse/mae) given a batch of inputs/outputs/targets.

        Args:
            x (torch.Tensor): Model inputs.
            y_hat (torch.Tensor): Model predictions.
            y (torch.Tensor): Model regression targets.
        """
        raise NotImplementedError("Should be implemented by subclass.")
