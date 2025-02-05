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
from probcal.training.losses import neg_binom_nll
from probcal.utils.differentiable_samplers import get_differentiable_sample_from_nbinom


class NegBinomNN(ProbabilisticRegressionNN):
    """A neural network that learns the parameters of a Negative Binomial distribution over each regression target (conditioned on the input).

    The mean-scale (mu, alpha) parametrization of the Negative Binomial is used for this network.

    Attributes:
        backbone (Backbone): Backbone to use for feature extraction.
        loss_fn (Callable): The loss function to use for training this NN.
        optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
        optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
        lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing". Defaults to None.
        lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}. Defaults to None.
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
        """Instantiate a NegBinomNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(NegBinomNN, self).__init__(
            loss_fn=neg_binom_nll,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 2),
            nn.Softplus(),  # To ensure positivity of output params.
        )

        self.nll = BootStrapper(NLL())
        self.mp = BootStrapper(MedianPrecision())
        self.crps = BootStrapper(ContinuousRankedProbabilityScore(mode="discrete"))
        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, alpha), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        h = self.backbone(x)
        y_hat = self.head(h)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 2).

        If viewing outputs as (mu, alpha), use `torch.split(y_hat, [1, 1], dim=-1)` to separate.
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)
        self.backbone.train()
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
        mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
        sample = (
            get_differentiable_sample_from_nbinom(
                mu=mu,
                alpha=alpha,
                num_samples=num_samples,
                temperature=kwargs.get("temperature", 0.1),
            )
            .squeeze(-1)
            .permute(1, 0)
        )
        return sample

    def _predictive_dist_impl(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.NegativeBinomial:
        mu, alpha = torch.split(y_hat, [1, 1], dim=-1)
        mu = mu.flatten()
        alpha = alpha.flatten()

        # Convert to standard parametrization.
        eps = torch.tensor(1e-6, device=mu.device)
        var = mu + alpha * mu**2
        p = mu / torch.maximum(var, eps)
        failure_prob = torch.minimum(
            1 - p, 1 - eps
        )  # Torch docs lie and say this should be P(success).
        n = mu**2 / torch.maximum(var - mu, eps)
        dist = torch.distributions.NegativeBinomial(total_count=n, probs=failure_prob)
        return dist

    def _point_prediction_impl(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        dist = self.predictive_dist(y_hat, training)
        return dist.mode

    def _update_addl_test_metrics(self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        dist = self.predictive_dist(y_hat)
        var = dist.variance
        precision = 1 / var
        targets = y.flatten()
        support = torch.arange(2000, device=y_hat.device).view(-1, 1)
        probs_over_support = torch.exp(dist.log_prob(support)).T

        self.nll.update(-dist.log_prob(targets))
        self.mp.update(precision)
        self.crps.update(probs_over_support, targets)

    def _log_addl_test_metrics(self):
        for name, metric in zip(("nll", "mp", "crps"), (self.nll, self.mp, self.crps)):
            result = metric.compute()
            self.log(f"{name}_mean", result["mean"])
            self.log(f"{name}_std", result["std"])
