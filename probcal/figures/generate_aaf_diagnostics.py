import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from torch.nn.functional import poisson_nll_loss
from tqdm import tqdm

from probcal.data_modules import AAFDataModule
from probcal.evaluation.kernels import polynomial_kernel
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.models import DoublePoissonNN
from probcal.models import FaithfulGaussianNN
from probcal.models import GaussianNN
from probcal.models import NaturalGaussianNN
from probcal.models import NegBinomNN
from probcal.models import PoissonNN
from probcal.models.probabilistic_regression_nn import ProbabilisticRegressionNN
from probcal.training.losses import double_poisson_nll
from probcal.training.losses import faithful_gaussian_nll
from probcal.training.losses import gaussian_nll
from probcal.training.losses import natural_gaussian_nll
from probcal.training.losses import neg_binom_nll


def embed_data_in_2d(
    train_embeddings: torch.Tensor, val_embeddings: torch.Tensor, test_embeddings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    all_embeddings_numpy = (
        torch.cat([train_embeddings, val_embeddings, test_embeddings]).detach().numpy()
    )
    all_embeddings_2d = TSNE(random_state=1998).fit_transform(all_embeddings_numpy)
    train_embeddings_2d = torch.tensor(all_embeddings_2d[: len(train_embeddings)])
    test_embeddings_2d = torch.tensor(all_embeddings_2d[-len(test_embeddings) :])
    return train_embeddings_2d, test_embeddings_2d


def draw_mcmd_samples_and_compute_losses(
    model: ProbabilisticRegressionNN,
    datamodule: L.LightningDataModule,
    val_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if isinstance(model, DoublePoissonNN):
        loss_fn = double_poisson_nll
    elif isinstance(model, GaussianNN):
        loss_fn = gaussian_nll
    elif isinstance(model, FaithfulGaussianNN):
        loss_fn = faithful_gaussian_nll
    elif isinstance(model, NaturalGaussianNN):
        loss_fn = natural_gaussian_nll
    elif isinstance(model, PoissonNN):
        loss_fn = poisson_nll_loss
    elif isinstance(model, NegBinomNN):
        loss_fn = neg_binom_nll

    y_hat_val = []
    y_val = []
    for image_batch, label_batch in tqdm(
        datamodule.val_dataloader(), desc="Sampling from predictive distributions..."
    ):
        y_hat = model.predict(image_batch)
        y_val.append(label_batch)
        y_hat_val.append(y_hat)

    losses = []
    for image_batch, label_batch in tqdm(datamodule.test_dataloader(), desc="Computing losses..."):
        y_hat = model.predict(image_batch)
        losses.append(loss_fn(y_hat, label_batch.float().unsqueeze(1)).item())
    losses = torch.tensor(losses)

    y_hat_val = torch.cat(y_hat_val)
    y_val = torch.cat(y_val)

    x = val_embeddings
    y = y_val
    x_prime = val_embeddings
    y_prime = model.sample(y_hat_val).flatten()

    return x, y, x_prime, y_prime, losses


def create_training_data_density_plot(train_embeddings_2d: torch.Tensor, save_path: Path):
    min_x, max_x = (
        train_embeddings_2d[:, 0].detach().numpy().min(),
        train_embeddings_2d[:, 0].detach().numpy().max(),
    )
    min_y, max_y = (
        train_embeddings_2d[:, 1].detach().numpy().min(),
        train_embeddings_2d[:, 1].detach().numpy().max(),
    )
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax: plt.Axes
    ax.set_title("Training Data Density")
    kde = gaussian_kde(train_embeddings_2d.T)
    grid_x, grid_y = np.mgrid[min_x:max_x:300j, min_y:max_y:300j]
    grid_densities = kde(np.vstack([grid_x.flatten(), grid_y.flatten()])).reshape(grid_x.shape)
    ax.contourf(grid_x, grid_y, grid_densities, levels=10, cmap="viridis")
    fig.savefig(save_path, dpi=150)


def get_meshgrid_test_embeddings(
    test_embeddings_2d: torch.Tensor, granularity: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    min_x, max_x = (
        test_embeddings_2d[:, 0].detach().numpy().min(),
        test_embeddings_2d[:, 0].detach().numpy().max(),
    )
    min_y, max_y = (
        test_embeddings_2d[:, 1].detach().numpy().min(),
        test_embeddings_2d[:, 1].detach().numpy().max(),
    )
    grid_x, grid_y = np.mgrid[min_x : max_x : granularity * 1j, min_y : max_y : granularity * 1j]
    return grid_x, grid_y


def create_test_target_plot(
    test_embeddings_2d: torch.Tensor,
    meshgrid: tuple[np.ndarray, np.ndarray],
    targets: torch.Tensor,
    save_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax: plt.Axes
    ax.set_title("Test Data (projected to 2d)")
    grid_data = griddata(test_embeddings_2d, targets.detach().numpy(), meshgrid, method="linear")
    mappable = ax.contourf(*meshgrid, grid_data, levels=5, cmap="viridis")
    fig.colorbar(mappable, ax=ax)
    fig.savefig(save_path, dpi=150)


def create_cce_plot(
    test_embeddings_2d: torch.Tensor,
    meshgrid: tuple[np.ndarray, np.ndarray],
    cce_vals: torch.Tensor,
    save_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax: plt.Axes
    ax.set_title(rf"$\overline{{\mathrm{{CCE}}}}$: {cce_vals.mean():.4f}")
    grid_cce = griddata(test_embeddings_2d, cce_vals, meshgrid, method="linear")
    mappable = ax.contourf(*meshgrid, grid_cce, levels=5, cmap="viridis")
    fig.colorbar(mappable, ax=ax)
    fig.savefig(save_path, dpi=150)


def create_loss_plot(
    test_embeddings_2d: torch.Tensor,
    meshgrid: tuple[np.ndarray, np.ndarray],
    losses: torch.Tensor,
    save_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax: plt.Axes
    ax.set_title("Test NLL")
    losses_grid = griddata(test_embeddings_2d, losses, meshgrid, method="linear")
    mappable = ax.contourf(*meshgrid, losses_grid, levels=5, cmap="viridis")
    fig.colorbar(mappable, ax=ax)
    fig.savefig(save_path, dpi=150)


def save_cce_best_worst_examples(
    datamodule: AAFDataModule, cce_vals: torch.Tensor, n: int, save_folder: Path
):
    cce_low_to_high = torch.argsort(cce_vals)
    least_aligned = cce_low_to_high[-n:]
    most_aligned = cce_low_to_high[:n]

    for i, img_idx in enumerate(least_aligned):
        cce = cce_vals[img_idx]
        img: Image.Image = datamodule.test.base[img_idx][0]
        img = img.resize((224, 224))
        img.save(save_folder / f"cce_worst_{i}_{cce:.4f}.jpg")
    for i, img_idx in enumerate(most_aligned):
        cce = cce_vals[img_idx]
        img: Image.Image = datamodule.test.base[img_idx][0]
        img = img.resize((224, 224))
        img.save(save_folder / f"cce_best_{i}_{cce:.4f}.jpg")


@torch.inference_mode()
def main(
    model: ProbabilisticRegressionNN,
    embeddings_dir: Path,
    save_folder: Path,
    num_highest_lowest_cce: int = 5,
):

    if not save_folder.exists():
        os.makedirs(save_folder)

    datamodule = AAFDataModule(
        root_dir="data/aaf", batch_size=1, num_workers=0, persistent_workers=False
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_embeddings = torch.load(embeddings_dir / "train_embeddings.pt", weights_only=True)
    val_embeddings = torch.load(embeddings_dir / "val_embeddings.pt", weights_only=True)
    test_embeddings = torch.load(embeddings_dir / "test_embeddings.pt", weights_only=True)

    print("Running TSNE to project data to 2d...")
    train_embeddings_2d, test_embeddings_2d = embed_data_in_2d(
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
    )
    print("Running inference and sampling from predictive distributions...")
    x, y, x_prime, y_prime, losses = draw_mcmd_samples_and_compute_losses(
        model=model,
        datamodule=datamodule,
        val_embeddings=val_embeddings,
    )
    x_kernel = polynomial_kernel
    y_kernel = partial(rbf_kernel, gamma=(1 / (2 * y.float().var())).item())

    print("Computing CCE values across test set...")
    cce_vals = compute_mcmd_torch(
        grid=test_embeddings,
        x=x,
        y=y.float(),
        x_prime=x_prime,
        y_prime=y_prime.float(),
        x_kernel=x_kernel,
        y_kernel=y_kernel,
        lmbda=0.1,
    )

    test_targets = torch.tensor([x[-1] for x in datamodule.test_dataloader()])
    meshgrid = get_meshgrid_test_embeddings(test_embeddings_2d, granularity=50)

    create_training_data_density_plot(
        train_embeddings_2d, save_path=save_folder / "training_densities.pdf"
    )
    create_test_target_plot(
        test_embeddings_2d,
        meshgrid,
        targets=test_targets,
        save_path=save_folder / "test_targets.pdf",
    )
    create_cce_plot(test_embeddings_2d, meshgrid, cce_vals, save_path=save_folder / "test_cce.pdf")
    create_loss_plot(test_embeddings_2d, meshgrid, losses, save_path=save_folder / "test_loss.pdf")
    save_cce_best_worst_examples(
        datamodule,
        cce_vals,
        n=num_highest_lowest_cce,
        save_folder=save_folder,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["ddpn", "faithful", "natural", "gaussian", "seitzer", "poisson", "nbinom"],
    )
    parser.add_argument("--model-ckpt", type=str)
    parser.add_argument(
        "--save-dir", type=str, default="probcal/figures/artifacts/aaf-diagnostics"
    )
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    if args.model_type == "ddpn":
        constructor = DoublePoissonNN
    elif args.model_type == "faithful":
        constructor = FaithfulGaussianNN
    elif args.model_type == "natural":
        constructor = NaturalGaussianNN
    elif args.model_type in ("gaussian", "seitzer"):
        constructor = GaussianNN
    elif args.model_type == "poisson":
        constructor = PoissonNN
    elif args.model_type == "nbinom":
        constructor = NegBinomNN
    else:
        raise ValueError("Invalid model type.")

    main(
        model=constructor.load_from_checkpoint(args.model_ckpt),
        embeddings_dir=Path("weights/embeddings/aaf"),
        save_folder=Path(args.save_dir) / args.model_type,
        num_highest_lowest_cce=args.n,
    )
