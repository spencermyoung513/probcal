import argparse
import math
import os
from functools import partial

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.loggers import CSVLogger
from scipy.stats import norm

from probcal.enums import DatasetType
from probcal.enums import LRSchedulerType
from probcal.enums import OptimizerType
from probcal.evaluation.kernels import laplacian_kernel
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.probabilistic_evaluator import ProbabilisticEvaluator
from probcal.evaluation.probabilistic_evaluator import ProbabilisticEvaluatorSettings
from probcal.figures.generate_cce_synthetic_figure import produce_figure
from probcal.models import GaussianNN
from probcal.models.backbones import MLP
from probcal.models.regression_nn import RegressionNN
from probcal.regularized_gaussian import RegularizedGaussianNN
from probcal.utils.experiment_utils import fix_random_seed
from probcal.utils.experiment_utils import get_chkp_callbacks
from probcal.utils.experiment_utils import get_datamodule

# ------------------------------------ Global Variables ------------------------------------#
BACKBONE_TYPE = MLP
BACKBONE_KWARGS = {"input_dim": 1}
OPTIM_TYPE = OptimizerType.ADAM_W
OPTIM_KWARGS = {"lr": 0.001, "weight_decay": 0.00001}
LR_SCHEDULER_TYPE = LRSchedulerType.COSINE_ANNEALING
LR_SCHEDULER_KWARGS = {"T_max": 200, "eta_min": 0, "last_epoch": -1}
DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------ Training Function ------------------------------------#
def train_model(
    ModelClass: RegressionNN,
    chkp_dir: str,
    datamodule: L.LightningDataModule,
    lmbda: float = None,
    kernel=rbf_kernel,
):

    fix_random_seed(1998)

    if ModelClass == GaussianNN:
        model = ModelClass(
            backbone_type=BACKBONE_TYPE,
            backbone_kwargs=BACKBONE_KWARGS,
            optim_type=OPTIM_TYPE,
            optim_kwargs=OPTIM_KWARGS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
        )
    else:
        model = ModelClass(
            backbone_type=BACKBONE_TYPE,
            backbone_kwargs=BACKBONE_KWARGS,
            optim_type=OPTIM_TYPE,
            optim_kwargs=OPTIM_KWARGS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
            lmbda=lmbda,
            kernel=kernel,
        )

    chkp_freq = 200
    log_dir = "logs"
    experiment_name = "cce_reg"
    chkp_callbacks = get_chkp_callbacks(chkp_dir, chkp_freq)
    logger = CSVLogger(save_dir=log_dir, name=experiment_name)

    num_epochs = 200
    trainer = L.Trainer(
        accelerator=DEVICE,
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


def gen_model_fit_plot(ModelClass, model_name, dataset_path, chkp_dir, lmbda):
    if ModelClass == GaussianNN:
        model = ModelClass.load_from_checkpoint(
            f"{chkp_dir}/best_loss.ckpt",
            backbone_type=BACKBONE_TYPE,
            backbone_kwargs=BACKBONE_KWARGS,
            optim_type=OPTIM_TYPE,
            optim_kwargs=OPTIM_KWARGS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
        )
    else:
        model = ModelClass.load_from_checkpoint(
            f"{chkp_dir}/best_loss.ckpt",
            backbone_type=BACKBONE_TYPE,
            backbone_kwargs=BACKBONE_KWARGS,
            optim_type=OPTIM_TYPE,
            optim_kwargs=OPTIM_KWARGS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
            lmbda=lmbda,
        )

    model = model.to(DEVICE)

    data = np.load(dataset_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    x_plot = np.linspace(X_test.min(), X_test.max(), 1000).reshape(-1, 1)
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_plot).to(DEVICE)

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

    plt.savefig(f"probcal/{model_name}_fit.png", dpi=300, bbox_inches="tight")
    plt.close()


def gen_cce_plot(ModelClasses, model_names, chkp_dir, lmbda):
    save_path = "probcal/gen_cce_plot.pdf"
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    models = [
        ModelClass.load_from_checkpoint(
            f"{chkp_dir}{model_name}/best_loss.ckpt",
            backbone_type=BACKBONE_TYPE,
            backbone_kwargs=BACKBONE_KWARGS,
            optim_type=OPTIM_TYPE,
            optim_kwargs=OPTIM_KWARGS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
            **({"lmbda": lmbda} if ModelClass != GaussianNN else {}),
        )
        for ModelClass, model_name in zip(ModelClasses, model_names)
    ]
    model_names = [model.replace("_", " ").capitalize() for model in model_names]
    produce_figure(models, model_names, save_path, dataset_path)


def compute_performance(
    ModelClass, model_name, kernel, kernel_name, chkp_dir, dataset_path, lmbda=None, batch_size: int = 32
):
    x_kernel = partial(kernel, gamma=0.5)

    settings = ProbabilisticEvaluatorSettings(
        dataset_type=DatasetType.TABULAR,
        cce_input_kernel=x_kernel,
        cce_output_kernel=kernel_name,
        cce_num_samples=1,
        cce_num_trials=1,
    )
    evaluator = ProbabilisticEvaluator(settings=settings)

    if ModelClass == GaussianNN:
        model = ModelClass.load_from_checkpoint(
            f"{chkp_dir}/best_loss.ckpt",
            backbone_type=BACKBONE_TYPE,
            backbone_kwargs=BACKBONE_KWARGS,
            optim_type=OPTIM_TYPE,
            optim_kwargs=OPTIM_KWARGS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
        )
    else:
        model = ModelClass.load_from_checkpoint(
            f"{chkp_dir}/best_loss.ckpt",
            backbone_type=BACKBONE_TYPE,
            backbone_kwargs=BACKBONE_KWARGS,
            optim_type=OPTIM_TYPE,
            optim_kwargs=OPTIM_KWARGS,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
            lmbda=lmbda,
        )

    dataset_type = DatasetType.TABULAR
    datamodule = get_datamodule(dataset_type, dataset_path, batch_size)

    results = evaluator(model, datamodule)
    cce = results.cce_results[0].mean_cce
    ece = results.ece

    # NLL
    data: dict[str, np.ndarray] = np.load(dataset_path)
    X = data["X_test"].flatten()
    y = data["y_test"].flatten()
    y_hat = model.predict(torch.tensor(X).unsqueeze(1))
    mu, var = torch.split(y_hat, [1, 1], dim=-1)
    mu = mu.flatten().detach().numpy()
    std = var.sqrt().flatten().detach().numpy()
    dist = norm(loc=mu, scale=std)
    nll = np.mean(-np.log(dist.cdf(y + 0.5) - dist.cdf(y - 0.5)))

    # MAE
    mae = np.mean(abs(y - mu))

    result = {
        "kernel": kernel_name,
        "model": model_name,
        "lambda": lmbda,
        "cce": cce,
        "ece": ece,
        "nll": nll,
        "mae": mae,
    }
    return result


def gen_kernel_cce_plot():
    df = pd.read_csv("experiment3.csv", index_col=0)

    df = df[df["model"] != "gaussian"]

    df["lambda"] = df["lambda"].astype(str)

    rbf = df[df["kernel"] == "rbf"]
    laplacian = df[df["kernel"] == "laplacian"]

    plt.title("ECE/CCE for Kernel/Lambda Combinations")
    plt.xlabel("Lambda")
    plt.ylabel("ECE/CCE")
    plt.plot(rbf["lambda"], rbf["cce"], label="RBF CCE")
    plt.plot(rbf["lambda"], rbf["ece"], label="RBF ECE")
    plt.plot(laplacian["lambda"], laplacian["cce"], label="Laplacian CCE")
    plt.plot(laplacian["lambda"], laplacian["ece"], label="Laplcaian ECE")
    plt.legend()
    plt.savefig("probcal/kernel_plot.png")
    plt.clf()


def gen_kernel_nll_plot():
    df = pd.read_csv("experiment3.csv", index_col=0)

    df = df[df["model"] != "gaussian"]

    df["lambda"] = df["lambda"].astype(str)

    rbf = df[df["kernel"] == "rbf"]
    laplacian = df[df["kernel"] == "laplacian"]

    plt.title("NLL for Kernel/Lambda Combinations")
    plt.xlabel("Lambda")
    plt.ylabel("NLL")
    plt.plot(rbf["lambda"], rbf["nll"], label="RBF")
    plt.plot(laplacian["lambda"], laplacian["nll"], label="Laplcaian")
    plt.legend()
    plt.savefig("probcal/nll_plot.png")
    plt.clf()


def gen_kernel_plot():
    df = pd.read_csv("experiment3.csv", index_col=0)

    df = df[df["model"] != "gaussian"]

    df["lambda"] = df["lambda"].astype(str)

    rbf = df[df["kernel"] == "rbf"]
    laplacian = df[df["kernel"] == "laplacian"]

    plt.title("MAE for Kernel/Lambda Combinations")
    plt.xlabel("Lambda")
    plt.ylabel("MAE")
    plt.plot(rbf["lambda"], rbf["mae"], label="RBF")
    plt.plot(laplacian["lambda"], laplacian["mae"], label="Laplcaian")
    plt.legend()
    plt.savefig("probcal/mae_plot.png")
    plt.clf()


def experiment_4_plot():
    df = pd.read_csv("experiment_4.csv", index_col=0)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.suptitle("Batch Size Comparison")
    plt.semilogx(df["batch_size"], df["ece"], "r-", label="ECE")
    plt.semilogx(df["batch_size"], df["cce"], "b-", label="CCE")
    plt.xlabel("Batch Size")
    plt.legend()
    plt.subplot(122)
    plt.semilogx(df["batch_size"], df["nll"], "g-", label="NLL")
    plt.xlabel("Batch Size")
    plt.legend()
    plt.savefig("experiment4.png")


def experiment_5_plot():
    df = pd.read_csv("experiment_5.csv")
    pivot_table = df.pivot(index="batch_size", columns="lambda", values="cce")
    # Avegae out the rows and columns
    pivot_table["Row Avg"] = pivot_table.mean(axis=1)
    pivot_table.loc["Column Avg"] = pivot_table.mean(axis=0)
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap="flare", fmt=".4f", cbar_kws={"label": "CCE"})
    plt.title("Lambda vs Batch Size")
    plt.xlabel("Lambda")
    plt.ylabel("Batch Size")
    plt.tight_layout()
    plt.show()


def experiment_1():
    ModelClasses = [RegularizedGaussianNN, GaussianNN]
    model_names = ["regularized_gaussian", "gaussian"]
    dataset_type = DatasetType.TABULAR
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    datamodule = get_datamodule(dataset_type, dataset_path, batch_size=32)

    for ModelClass, model_name in zip(ModelClasses, model_names):
        chkp_dir = "chkp/experiment1/" + model_name

        if not os.path.exists(chkp_dir):
            train_model(ModelClass, chkp_dir, datamodule, lmbda=0.01)
        gen_model_fit_plot(ModelClass, model_name, dataset_path, chkp_dir, lmbda=0.01)


def experiment_2():
    ModelClasses = [GaussianNN, RegularizedGaussianNN]
    model_names = ["gaussian", "regularized_gaussian"]
    chkp_dir_base = "chkp/experiment2/"
    dataset_type = DatasetType.TABULAR
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    datamodule = get_datamodule(dataset_type, dataset_path, 32)

    for ModelClass, model_name in zip(ModelClasses, model_names):
        chkp_dir = chkp_dir_base + model_name
        if not os.path.exists(chkp_dir):
            train_model(ModelClass, chkp_dir, datamodule, lmbda=0.1)

    gen_cce_plot(ModelClasses, model_names, chkp_dir_base, lmbda=0.1)


def experiment_3():
    kernels = [rbf_kernel, laplacian_kernel]
    kernel_names = ["rbf", "laplacian"]
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100]  # np.linspace(0, 1, 20)
    model_names = ["gaussian", "regularized_gaussian"]
    ModelClasses = [GaussianNN, RegularizedGaussianNN]
    dataset_type = DatasetType.TABULAR
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"

    datamodule = get_datamodule(dataset_type, dataset_path, 32)

    results = []
    for kernel_name, kernel in zip(kernel_names, kernels):
        for model_name, ModelClass in zip(model_names, ModelClasses):

            # Train regularized models
            if model_name == "regularized_gaussian":
                for lmbda in lambdas:
                    chkp_dir = (
                        "chkp/experiment3/" + kernel_name + "_" + model_name + "_" + str(lmbda)
                    )

                    if not os.path.exists(f"{chkp_dir}/last.ckpt"):
                        print(
                            f"Training {model_name} with lambda={lmbda} and kernel={kernel_name}"
                        )
                        train_model(ModelClass, chkp_dir, datamodule, lmbda, kernel)
                    result = compute_performance(
                        ModelClass, model_name, kernel, kernel_name, chkp_dir, dataset_path, lmbda
                    )
                    results.append(result)

            # Train standard gaussian models
            else:
                chkp_dir = "chkp/experiment3/" + kernel_name + "_" + model_name

                if not os.path.exists(f"{chkp_dir}/last.ckpt"):
                    print(f"Training {model_name} with kernel={kernel_name}")
                    train_model(ModelClass, chkp_dir, datamodule)
                result = compute_performance(ModelClass, model_name, kernel, kernel_name, chkp_dir, dataset_path)
                results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("experiment3.csv")
    gen_kernel_cce_plot()
    gen_kernel_nll_plot()
    gen_kernel_plot()


def experiment_4():
    batch_sizes = [2**i for i in range(2, 11)]
    results = []
    dataset_type = DatasetType.TABULAR
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    
    for batch in batch_sizes:
        chkp_dir = "chkp/experiment4/batch_" + str(batch)
        if not os.path.exists(f"{chkp_dir}/last.ckpt"):
            print(f"Training with batch = {batch}")
            datamodule = get_datamodule(dataset_type, dataset_path, batch)
            train_model(RegularizedGaussianNN, chkp_dir, datamodule, lmbda=0.1)
        result = compute_performance(
            ModelClass=RegularizedGaussianNN,
            model_name="regularized_gaussian",
            kernel=laplacian_kernel,
            kernel_name="laplacian",
            chkp_dir=chkp_dir,
            dataset_path=dataset_path,
            batch_size=batch,
            lmbda=0.1,
        )
        result["batch_size"] = batch
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("experiment_4.csv")
    experiment_4_plot()


def experiment_5():
    dataset_type = DatasetType.TABULAR
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    results = []
    batch_sizes = [2**i for i in range(2, 10)] + [800]
    lmbdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    

    for batch in batch_sizes:
        datamodule = get_datamodule(dataset_type, dataset_path, batch)
        for lmbda in lmbdas:
            chkp_dir = f"chkp/batch_gamma_experiments/batch_{batch}_gamma_{lmbda}"
            if not os.path.exists(f"{chkp_dir}/last.ckpt"):
                print(f"Training with batch = {batch}, lambda = {lmbda}")
                train_model(RegularizedGaussianNN, chkp_dir, datamodule,  lmbda=lmbda)
            print(f"Evaluating with batch = {batch}, lambda = {lmbda}")
            result = compute_performance(
                ModelClass=RegularizedGaussianNN,
                model_name="regularized_gaussian",
                kernel=laplacian_kernel,
                kernel_name="laplacian",
                chkp_dir=chkp_dir,
                dataset_path=dataset_path,
                batch_size=batch,
                lmbda=lmbda,
            )
            result["batch_size"] = batch
            results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("experiment_5.csv")

def experiment_6():
    dataset_type = DatasetType.IMAGE
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    results = []
    batch_sizes = [2**i for i in range(5, 10)]
    lmbdas = [0.001, 0.01, 0.1, 1, 10]

    for batch in batch_sizes:
        datamodule = get_datamodule(dataset_type, dataset_path, batch)
        for lmbda in lmbdas:
            chkp_dir = f"chkp/experiment_6/batch_{batch}_lmbda_{lmbda}"
            if not os.path.exists(f"{chkp_dir}/last.ckpt"):
                print(f"Training with batch = {batch}, lambda = {lmbda}")

                train_model(RegularizedGaussianNN, chkp_dir, batch_size=batch, datamodule=datamodule, lmbda=lmbda)
            print(f"Evaluating with batch = {batch}, lambda = {lmbda}")
            result = compute_performance(
                ModelClass=RegularizedGaussianNN,
                model_name="regularized_gaussian",
                kernel=laplacian_kernel,
                kernel_name="laplacian",
                chkp_dir=chkp_dir,
                dataset_path=dataset_path,
                batch_size=batch,
                lmbda=lmbda,
            ) 
            result["batch_size"] = batch
            results.append(result)
    df = pd.DataFrame(results)
    df.to_csv("experiment_6.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int)
    args = parser.parse_args()

    match args.experiment:

        case 1:
            experiment_1()

        case 2:
            experiment_2()

        case 3:
            experiment_3()

        case 4:
            experiment_4()

        case 5:
            experiment_5()
        
        case 6:
            experiment_6()
