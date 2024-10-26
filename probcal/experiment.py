import math
import os
import warnings
from functools import partial
from typing import Optional
from typing import Type

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from probcal.enums import DatasetType
from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.models import GaussianNN
from probcal.models.backbones import MLP
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_chkp_callbacks
from probcal.utils.experiment_utils import get_datamodule

from probcal.evaluation.probabilistic_evaluator import ProbabilisticEvaluator, ProbabilisticEvaluatorSettings
from probcal.enums import BetaSchedulerType
from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.evaluation.custom_torchmetrics import AverageNLL
from probcal.models.backbones import Backbone
from probcal.models.regression_nn import RegressionNN
from probcal.training.beta_schedulers import CosineAnnealingBetaScheduler
from probcal.training.beta_schedulers import LinearBetaScheduler
from probcal.evaluation.kernels import rbf_kernel

BACKBONE_TYPE = MLP
BACKBONE_KWARGS = {"input_dim": 1}
OPTIM_TYPE = OptimizerType.ADAM_W
OPTIM_KWARGS = {"lr": 0.001, "weight_decay": 0.00001}
LR_SCHEDULER_TYPE = LRSchedulerType.COSINE_ANNEALING
LR_SCHEDULER_KWARGS = {"T_max": 200, "eta_min": 0, "last_epoch": -1}

DATASET_PATH = "data/discrete_sine_wave/discrete_sine_wave.npz"

CHKP_DIR = "chkp/gauss_reg"


def train_model(ModelClass: RegressionNN):

    fix_random_seed(1124)

    dataset_type = DatasetType.TABULAR
    batch_size = 32
    datamodule = get_datamodule(dataset_type, DATASET_PATH, batch_size)

    model = ModelClass(
        backbone_type=BACKBONE_TYPE,
        backbone_kwargs=BACKBONE_KWARGS,
        optim_type=OPTIM_TYPE,
        optim_kwargs=OPTIM_KWARGS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
    )

    chkp_freq = 25
    log_dir = "logs"
    experiment_name = "cce_reg"
    chkp_callbacks = get_chkp_callbacks(CHKP_DIR, chkp_freq)
    logger = CSVLogger(save_dir=log_dir, name=experiment_name)

    num_epochs = 100
    trainer = L.Trainer(
        accelerator="cpu",
        min_epochs=num_epochs,
        max_epochs=num_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=math.ceil(num_epochs / 200),
        enable_model_summary=False,
        callbacks=chkp_callbacks,
        logger=logger,
        precision=None,
    )
    trainer.fit(model=model, datamodule=datamodule)


def generate_figure(ModelClass):

    model = ModelClass.load_from_checkpoint(
        f"{CHKP_DIR}/last.ckpt",
        backbone_type=BACKBONE_TYPE,
        backbone_kwargs=BACKBONE_KWARGS,
        optim_type=OPTIM_TYPE,
        optim_kwargs=OPTIM_KWARGS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data = np.load(DATASET_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    x_plot = np.linspace(X_test.min(), X_test.max(), 1000).reshape(-1, 1)
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_plot).to(device)

        output = model(x_tensor)
        output = output.cpu().numpy()

        pred_mean = output[:, 0]
        pred_std = np.exp(0.5 * output[:, 1])  # Convert log variance to standard deviation

    plt.figure(figsize=(10, 6))

    plt.scatter(X_train, y_train, color="blue", alpha=0.5, label="Training Data", s=20)

    plt.scatter(X_test, y_test, color="red", alpha=0.5, label="Test Data", s=20)

    plt.plot(x_plot, pred_mean, color="green", label="Predicted Mean")

    plt.fill_between(
        x_plot.flatten(),
        pred_mean.flatten() - 2 * pred_std.flatten(),
        pred_mean.flatten() + 2 * pred_std.flatten(),
        color="green",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Discrete Sine Wave: True Data vs Model Predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("probcal/experiment_results.png", dpi=300, bbox_inches="tight")
    plt.close()


def gaussian_nll_cce(
    model: GaussianNN, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor, beta: float | None = None, lmbda: float = 0.1,
) -> torch.Tensor:

    if targets.size(1) != 1:
        warnings.warn(
            f"Targets tensor for `gaussian_nll` expected to be of shape (n, 1) but got shape {targets.shape}. This may result in unexpected training behavior."
        )
    if beta is not None:
        if beta < 0 or beta > 1:
            raise ValueError(f"Invalid value of beta specified. Must be in [0, 1]. Got {beta}")
    
    mu, logvar = torch.split(outputs, [1, 1], dim=-1)
    sample_dataset = TensorDataset(inputs, mu)
    grid_dataset = TensorDataset(inputs, targets)

    sample_loader = DataLoader(sample_dataset, batch_size=64, shuffle=True)
    grid_loader = DataLoader(grid_dataset, batch_size=64, shuffle=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    x_vals = torch.cat([x for x, _ in sample_loader], dim=0)
    gamma = (1 / (2 * x_vals.var())).item()
    cce_input_kernel = partial(rbf_kernel, gamma=gamma)

    prob_eval_settings = ProbabilisticEvaluatorSettings(
        dataset_type=DatasetType.TABULAR,
        device=device,
        cce_num_trials=5,
        cce_input_kernel=cce_input_kernel,
        cce_output_kernel="rbf",
        cce_lambda=.1,
        cce_num_samples=1,
        ece_bins=50,
        ece_weights="frequency",
        ece_alpha=1.0
    )

    nll = 0.5 * (torch.exp(-logvar) * (targets - mu) ** 2 + logvar)
    evaluator = ProbabilisticEvaluator(prob_eval_settings)
    cce_vals  = evaluator.compute_cce(
        model=model,
        grid_loader=grid_loader,
        sample_loader=sample_loader,
        complex_inputs=False
    )
    mean_cce = cce_vals.mean().item()
    
    losses = nll + lmbda*mean_cce

    if beta is not None and beta != 0:
        var = torch.exp(logvar)
        losses = torch.pow(var.detach(), beta) * losses

    return losses.mean()


class RegularizedGaussianNN(GaussianNN):

    def __init__(
        self,
        backbone_type: Type[Backbone],
        backbone_kwargs: dict,
        optim_type: Optional[OptimizerType] = None,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_type: Optional[LRSchedulerType] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        beta_scheduler_type: BetaSchedulerType | None = None,
        beta_scheduler_kwargs: dict | None = None,
    ):
        
        if beta_scheduler_type == BetaSchedulerType.COSINE_ANNEALING:
            self.beta_scheduler = CosineAnnealingBetaScheduler(**beta_scheduler_kwargs)
        elif beta_scheduler_type == BetaSchedulerType.LINEAR:
            self.beta_scheduler = LinearBetaScheduler(**beta_scheduler_kwargs)
        else:
            self.beta_scheduler = None

        super(GaussianNN, self).__init__(
            loss_fn=partial(
                gaussian_nll_cce,
                beta=(
                    self.beta_scheduler.current_value if self.beta_scheduler is not None else None
                ),
            ),
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
        loss = self.loss_fn(model=self, inputs=x.view(-1, 1).float(), outputs=y_hat, targets=y.view(-1, 1).float())
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
        loss = self.loss_fn(model=self, inputs=x.view(-1, 1).float(), outputs=y_hat, targets=y.view(-1, 1).float())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Since we used the model's forward method, we specify training=True to get the proper transforms.
        point_predictions = self.point_prediction(y_hat, training=True).flatten()
        self.val_rmse.update(point_predictions, y.flatten().float())
        self.val_mae.update(point_predictions, y.flatten().float())
        self.log("val_rmse", self.val_rmse, on_epoch=True)
        self.log("val_mae", self.val_mae, on_epoch=True)

        return loss

if __name__ == "__main__":
    if not os.path.exists(f"{CHKP_DIR}/last.ckpt"):
        train_model(RegularizedGaussianNN)
    generate_figure(RegularizedGaussianNN)
