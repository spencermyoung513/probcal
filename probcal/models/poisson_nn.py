from functools import partial
from typing import Optional
from typing import Type

import torch
from torch import nn
from torch.nn.functional import poisson_nll_loss
from torchmetrics import Metric

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.evaluation.custom_torchmetrics import ContinuousRankedProbabilityScore
from probcal.evaluation.custom_torchmetrics import MedianPrecision
from probcal.models.backbones import Backbone
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.utils.differentiable_samplers import get_differentiable_sample_from_poisson


class PoissonNN(ProbabilisticRegressionNN):
    """A neural network that learns the parameters of a Poisson distribution over each regression target (conditioned on the input).

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
        optim_type: Optional[OptimizerType] = None,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_type: Optional[LRSchedulerType] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        """Instantiate a PoissonNN.

        Args:
            backbone_type (Type[Backbone]): Type of backbone to use for feature extraction (can be initialized with backbone_type()).
            backbone_kwargs (dict): Keyword arguments to instantiate the backbone.
            optim_type (OptimizerType): The type of optimizer to use for training the network, e.g. "adam", "sgd", etc.
            optim_kwargs (dict): Key-value argument specifications for the chosen optimizer, e.g. {"lr": 1e-3, "weight_decay": 1e-5}.
            lr_scheduler_type (LRSchedulerType | None): If specified, the type of learning rate scheduler to use during training, e.g. "cosine_annealing".
            lr_scheduler_kwargs (dict | None): If specified, key-value argument specifications for the chosen lr scheduler, e.g. {"T_max": 500}.
        """
        super(PoissonNN, self).__init__(
            loss_fn=partial(poisson_nll_loss, log_input=True),
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Linear(self.backbone.output_dim, 1)

        self.mp = MedianPrecision()
        self.crps = ContinuousRankedProbabilityScore(mode="discrete")

        self.save_hyperparameters()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).
        """
        h = self.backbone(x)
        y_hat = self.head(h)  # Interpreted as log(mu)
        return y_hat

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction with the network.

        Args:
            x (torch.Tensor): Batched input tensor with shape (N, ...).

        Returns:
            torch.Tensor: Output tensor, with shape (N, 1).
        """
        self.backbone.eval()
        y_hat = self._forward_impl(x)  # Interpreted as log(mu)
        self.backbone.train()

        return torch.exp(y_hat)

    def _sample_impl(
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
        sample = (
            get_differentiable_sample_from_poisson(
                lmbda=y_hat, num_samples=num_samples, temperature=kwargs.get("temperature", 0.1)
            )
            .squeeze(-1)
            .permute(1, 0)
        )
        return sample

    def _predictive_dist_impl(
        self, y_hat: torch.Tensor, training: bool = False
    ) -> torch.distributions.Poisson:
        lmbda = y_hat.exp() if training else y_hat
        dist = torch.distributions.Poisson(lmbda.squeeze())
        return dist

    def _point_prediction_impl(self, y_hat: torch.Tensor, training: bool) -> torch.Tensor:
        lmbda = y_hat.exp() if training else y_hat
        return lmbda.floor()

    def _addl_test_metrics_dict(self) -> dict[str, Metric]:
        return {
            "mp": self.mp,
            "crps": self.crps,
        }

    def _update_addl_test_metrics_batch(
        self, x: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ):
        lmbda = y_hat.flatten()
        targets = y.flatten()
        precision = 1 / lmbda
        support = torch.arange(2000, device=y_hat.device).view(-1, 1)
        dist = self.predictive_dist(y_hat, training=False)
        probs_over_support = torch.exp(dist.log_prob(support)).T

        self.mp.update(precision)
        self.crps.update(probs_over_support, targets)
