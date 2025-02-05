from typing import Type

import torch
from torch import nn
from torchmetrics import BootStrapper

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.evaluation.custom_torchmetrics import ContinuousRankedProbabilityScore
from probcal.evaluation.custom_torchmetrics import MedianPrecision
from probcal.evaluation.custom_torchmetrics import NLL
from probcal.models.backbones import Backbone
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.training.losses import faithful_gaussian_nll
from probcal.utils.differentiable_samplers import get_differentiable_sample_from_gaussian


class FaithfulGaussianNN(ProbabilisticRegressionNN):
    """Implementation of https://arxiv.org/abs/2212.09184.

    Attributes:
        backbone (Backbone): Backbone to use for feature extraction.
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
    """

    def __init__(
        self,
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: OptimizerType | None = None,
        optim_kwargs: dict | None = None,
        lr_scheduler_type: LRSchedulerType | None = None,
        lr_scheduler_kwargs: dict | None = None,
    ):
        """Instantiate a FaithfulGaussianNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(FaithfulGaussianNN, self).__init__(
            loss_fn=faithful_gaussian_nll,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.mu_head = nn.Linear(self.backbone.output_dim, 1)
        self.logvar_head = nn.Linear(self.backbone.output_dim, 1)

        self.nll = BootStrapper(NLL())
        self.mp = BootStrapper(MedianPrecision())
        self.crps = BootStrapper(ContinuousRankedProbabilityScore(mode="gaussian"))
        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, logvar), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h: torch.Tensor = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h.detach())
        y_hat = torch.cat((mu, logvar), dim=-1)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, var), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)
        self.backbone.train()

        # Apply torch.exp to the logvar dimension.
        output_shape = y_hat.shape
        reshaped = y_hat.view(-1, 2)
        y_hat = torch.stack([reshaped[:, 0], torch.exp(reshaped[:, 1])], dim=1).view(*output_shape)

        return y_hat

    def _sample_impl(
        self, y_hat: torch.Tensor, training: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        """Sample from this network's posterior predictive distributions for a batch of data (as specified by y_hat).

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.
            num_samples (int, optional): Number of samples to take from each posterior predictive. Defaults to 1.

        Returns:
            torch.Tensor: Batched sample tensor, with shape (N, num_samples).
        """
        dist = self.predictive_dist(y_hat, training)
        sample = dist.sample((num_samples,)).view(num_samples, -1).T
        return sample

    def _rsample_impl(
        self, y_hat: torch.Tensor, training: bool = False, num_samples: int = 1, **kwargs
    ) -> torch.Tensor:
        """Sample (using a differentiable relaxation) from this network's predictive distributions for a batch of data (as specified by y_hat).

        Args:
            y_hat (torch.Tensor): Output tensor from a regression network, with shape (N, ...).
            training (bool, optional): Boolean indicator specifying if `y_hat` is a training output or not. This particularly matters when outputs are in log space during training, for example. Defaults to False.
            num_samples (int, optional): Number of samples to take from each predictive distribution. Defaults to 1.

        Returns:
            torch.Tensor: Batched sample tensor, with shape (N, num_samples).
        """
        dist: torch.distributions.Normal = self.predictive_dist(y_hat, training)
        sample = (
            get_differentiable_sample_from_gaussian(
                mu=dist.loc,
                stdev=dist.scale,
                num_samples=num_samples,
            )
            .squeeze(-1)
            .permute(1, 0)
        )
        return sample

    def _predictive_dist_impl(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.Normal:
        if training:
            mu, logvar = torch.split(y_hat, [1, 1], dim=-1)
            var = logvar.exp()
        else:
            mu, var = torch.split(y_hat, [1, 1], dim=-1)

        dist = torch.distributions.Normal(loc=mu.squeeze(), scale=var.sqrt().squeeze())
        return dist

    def _point_prediction_impl(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        mu, _ = torch.split(y_hat, [1, 1], dim=-1)
        return mu

    def _update_addl_test_metrics(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        mu, var = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        var = var.flatten()
        precision = 1 / var
        targets = y.flatten()
        dist = self.predictive_dist(y_hat, training=False)

        self.nll.update(-dist.log_prob(targets))
        self.mp.update(precision)
        self.crps.update(y_hat, targets)

    def _log_addl_test_metrics(self):
        for name, metric in zip(("nll", "mp", "crps"), (self.nll, self.mp, self.crps)):
            result = metric.compute()
            self.log(f"{name}_mean", result["mean"])
            self.log(f"{name}_std", result["std"])
