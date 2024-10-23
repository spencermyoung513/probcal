import math
import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger

from probcal.enums import DatasetType
from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.models import GaussianNN
from probcal.models.backbones import MLP
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_chkp_callbacks
from probcal.utils.experiment_utils import get_datamodule

BACKBONE_TYPE = MLP
BACKBONE_KWARGS = {"input_dim": 1}
OPTIM_TYPE = OptimizerType.ADAM_W
OPTIM_KWARGS = {"lr": 0.001, "weight_decay": 0.00001}
LR_SCHEDULER_TYPE = LRSchedulerType.COSINE_ANNEALING
LR_SCHEDULER_KWARGS = {"T_max": 200, "eta_min": 0, "last_epoch": -1}

DATASET_PATH = "data/discrete_sine_wave/discrete_sine_wave.npz"

CHKP_DIR = "chkp/gauss_classic"


def train_model():

    fix_random_seed(1124)

    dataset_type = DatasetType.TABULAR
    batch_size = 32
    datamodule = get_datamodule(dataset_type, DATASET_PATH, batch_size)

    model = GaussianNN(
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
        accelerator="gpu",
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


def generate_figure():

    model = GaussianNN.load_from_checkpoint(
        f"{CHKP_DIR}/last.ckpt",
        backbone_type=BACKBONE_TYPE,
        backbone_kwargs=BACKBONE_KWARGS,
        optim_type=OPTIM_TYPE,
        optim_kwargs=OPTIM_KWARGS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
    )
    model.eval()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
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


if __name__ == "__main__":
    if not os.path.exists(f"{CHKP_DIR}/last.ckpt"):
        train_model()
    generate_figure()
