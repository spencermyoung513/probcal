import argparse
import logging
import os.path
from datetime import datetime
from functools import partial
from typing import Type

import matplotlib.pyplot as plt
import open_clip
import torch
from tqdm import tqdm

from probcal.enums import DatasetType
from probcal.enums import ImageDatasetName
from probcal.evaluation.kernels import polynomial_kernel
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.random_variables.double_poisson import DoublePoisson
from probcal.utils.configs import EvaluationConfig
from probcal.utils.experiment_utils import from_yaml
from probcal.utils.experiment_utils import get_datamodule
from probcal.utils.experiment_utils import get_model


def mk_log_dir(log_dir, exp_name):
    """
    Creates a directory for logging experiment results. Assumes that the log directory is a subdirectory of the current working directory.

    Args:
        log_dir (str): The base directory where logs should be stored.
        exp_name (str): The name of the experiment, which will be used to create a subdirectory within the base log directory.

    Returns:
        None: This function does not return a value but creates directories as needed.
    """
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M").replace(" ", "-")
    log_dir = os.path.join(log_dir, exp_name + "_" + ts)
    log_file = os.path.join(log_dir, "log.txt")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir, log_file


def get_y_kernel(Y_true: torch.Tensor, gamma: str | float):

    if gamma == "auto":
        return partial(rbf_kernel, gamma=1 / (2 * Y_true.float().var()))
    elif isinstance(gamma, float):
        return partial(rbf_kernel, gamma=gamma)
    else:
        raise ValueError(f"Invalid gamma value: {gamma}")


def main(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir, log_file = mk_log_dir(cfg["exp"]["log_dir"], cfg["exp"]["name"])
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    logging.info(f"Beginning experiment {cfg['exp']['name']}")
    logging.info(f"Model Config: {cfg['model']}")
    print(f"Getting models weights: {cfg['model']['weights']}")
    logging.info(f"Data Config: {cfg['data']}")
    logging.info(f"Hyperparam Config: {cfg['hyperparams']}")
    logging.info(f"Device for experiment: {device}")
    # build dataset and data loader
    datamodule = get_datamodule(
        DatasetType.IMAGE, ImageDatasetName(cfg["data"]["module"]), 1, num_workers=0
    )
    logging.info(f"DataModule: {type(datamodule)}")
    if cfg["data"]["perturb"] is None:
        datamodule.setup(stage="test")
    else:
        datamodule.setup(stage="test", perturb=cfg["data"]["perturb"])
    test_loader = datamodule.test_dataloader()

    # instantiate model
    model_cfg = EvaluationConfig.from_yaml(cfg["model"]["test_cfg"])
    # model = get_model(model_cfg).to(device)
    initializer: Type[ProbabilisticRegressionNN] = get_model(model_cfg, return_initializer=True)[1]
    model = initializer.load_from_checkpoint(model_cfg.model_ckpt_path)
    model = model.to(device)
    logging.info("Model loaded")
    logging.info(f"Model: {type(model)}")

    # get embeder
    embedder, _, transform = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
    )
    embedder.eval()

    n = cfg["data"]["test_examples"] if cfg["data"]["test_examples"] else len(test_loader)
    m = cfg["data"]["n_samples"]
    logging.info(f"Processing {n} test examples")
    logging.info(f"Sampling {m} times from model")
    X = torch.zeros((n, 512), device=device)  # image embeddings
    Y_true = torch.zeros((n, 1), device=device)  # true labels
    Y_prime = []  # sampled model outputs
    imgs_to_plot = []
    imgs_to_plot_preds = []
    imgs_to_plot_true = []

    for i, (x, y) in tqdm(enumerate(test_loader), total=n):
        with torch.no_grad():
            img_features = embedder.encode_image(x.to(device), normalize=False)
            pred = model._predict_impl(x.to(device))
            samples = model._sample_impl(pred, training=False, num_samples=m)

        X[i] = img_features
        Y_true[i] = y.to(device)
        Y_prime.append(samples.T)

        if i < cfg["plot"]["num_img_to_plot"]:
            img = datamodule.denormalize(x)
            img = img.squeeze(0).permute(1, 2, 0).detach()
            imgs_to_plot.append(img)
            imgs_to_plot_preds.append(pred)
            imgs_to_plot_true.append(y)

        if i == (n - 1):
            break

    # plot images
    fig, axs = plt.subplots(4, 2, figsize=(10, 8), sharey="col")
    imgs_to_plot_preds = torch.cat(imgs_to_plot_preds, dim=0)
    imgs_to_plot_true = torch.cat(imgs_to_plot_true, dim=0)
    for i in range(cfg["plot"]["num_img_to_plot"]):
        axs[i, 0].imshow(imgs_to_plot[i])
        axs[i, 0].set_title(f"Image: {i + 1}")
        axs[i, 0].axis("off")

        rv = model.predictive_dist(imgs_to_plot_preds[i], training=False)
        disc_support = torch.arange(0, imgs_to_plot_true.max() + 5)
        if isinstance(rv, DoublePoisson):
            dist_func = torch.exp(rv._logpmf(disc_support.to(device)))
        else:
            dist_func = torch.exp(rv.log_prob(disc_support.to(device)))
        axs[i, 1].plot(disc_support.cpu(), dist_func.cpu())
        axs[i, 1].scatter(imgs_to_plot_true[i], 0, color="black", marker="*", s=50, zorder=100)

    plt.savefig(os.path.join(log_dir, "input_images.png"))

    # free up memory allocated to models
    logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024.0 / 1024.0} MB")
    del model
    del embedder
    torch.cuda.empty_cache()
    logging.info("Model and embedder removed")
    logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024.0 / 1024.0} MB")

    # compute CCE
    Y_prime = torch.cat(Y_prime, dim=0).to(device)

    with torch.inference_mode():
        x_prime = X.repeat_interleave(m, dim=0).to(device)
        print(x_prime.shape, Y_prime.shape)

        cce_vals = compute_mcmd_torch(
            grid=X,
            x=X,
            y=Y_true.float(),
            x_prime=x_prime,
            y_prime=Y_prime.float(),
            x_kernel=polynomial_kernel,
            y_kernel=get_y_kernel(Y_true, cfg["hyperparams"]["y_kernel_gamma"]),
            lmbda=cfg["hyperparams"]["lmbda"],
        )

    # check if there are nan values
    if torch.isnan(cce_vals).any():
        logging.error("CCE values contain NaNs")
        # count the number of NaNs
        nans = torch.isnan(cce_vals).sum()
        logging.error(f"Number of NaNs: {nans}")
        # mask the NaNs
        cce_vals = cce_vals[~torch.isnan(cce_vals)]
    else:
        logging.info(f"CCE values: {cce_vals}")

    print(cce_vals.mean())
    logging.info(f"Final CCE: {cce_vals.mean()}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str)
    args = args.parse_args()

    cfg = from_yaml(args.config)
    try:
        main(cfg)
    except Exception as e:
        logging.exception(e)
