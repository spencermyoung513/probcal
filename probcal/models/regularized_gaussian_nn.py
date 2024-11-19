from functools import partial
from typing import Optional
from typing import Type

import open_clip
import torch
from torch import nn

from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.evaluation.custom_torchmetrics import AverageNLL
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.models import GaussianNN
from probcal.models.backbones import Backbone

clip_model, _, image_preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=torch.device("cuda")
)


def gaussian_nll_cce(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    kernel=rbf_kernel,
) -> torch.Tensor:

    with torch.no_grad:
        grid = clip_model.encode_image(inputs, normalize=False)

    mu, logvar = torch.split(outputs, [1, 1], dim=-1)
    stdev = (0.5 * logvar).exp()

    # Use the reparametrization trick to obtain 1 sample from the model's predictive distribution for each target.
    eps = torch.randn_like(stdev)
    y_prime = mu + eps * stdev
    x_gamma = (1 / (2 * inputs.var())).item()
    y_gamma = (1 / (2 * targets.var())).item()
    cce_vals = compute_mcmd_torch(
        grid=grid,
        x=grid,
        y=targets,
        x_prime=grid,
        y_prime=y_prime,
        x_kernel=partial(kernel, gamma=x_gamma),
        y_kernel=partial(kernel, gamma=y_gamma),
        lmbda=0.01,
    )

    nll = 0.5 * (torch.exp(-logvar) * (targets - mu) ** 2 + logvar)
    mean_cce = cce_vals.mean()

    loss = nll.mean() + alpha * mean_cce
    return loss


# ------------------------------------ GaussianNN With CCE Regularization ------------------------------------#


class RegularizedGaussianNN(GaussianNN):
    def __init__(
        self,
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: Optional[OptimizerType] = None,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_type: Optional[LRSchedulerType] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        alpha: float = 0.1,
        kernel=rbf_kernel,
    ):
        self.beta_scheduler = None

        super(GaussianNN, self).__init__(
            loss_fn=partial(gaussian_nll_cce, alpha=alpha, kernel=kernel),
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
            optim_type=optim_type,
            optim_kwargs=optim_kwargs,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.head = nn.Linear(self.backbone.output_dim, 2)
        self.nll = AverageNLL()
        self.save_hyperparameters()

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(inputs=x, outputs=y_hat, targets=y.view(-1, 1).float())
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

        loss = self.loss_fn(inputs=x, outputs=y_hat, targets=y.view(-1, 1).float())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Since we used the model's forward method, we specify training=True to get the proper transforms.
        point_predictions = self.point_prediction(y_hat, training=True).flatten()
        self.val_rmse.update(point_predictions, y.flatten().float())
        self.val_mae.update(point_predictions, y.flatten().float())
        self.log("val_rmse", self.val_rmse, on_epoch=True)
        self.log("val_mae", self.val_mae, on_epoch=True)

        return loss
