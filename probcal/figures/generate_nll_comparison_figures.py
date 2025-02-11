import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm

from probcal.data_modules import AAFDataModule
from probcal.evaluation.calibration_evaluator import CalibrationResults
from probcal.models import GaussianNN


def plot_cce_vs_nll(
    input_grid_2d: np.ndarray,
    targets: np.ndarray,
    cce_means: np.ndarray,
    nll_vals: np.ndarray,
    gridsize: int = 100,
    show: bool = False,
):

    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    axs: list[plt.Axes]
    grid_x, grid_y = np.mgrid[
        min(input_grid_2d[:, 0]) : max(input_grid_2d[:, 0]) : gridsize * 1j,
        min(input_grid_2d[:, 1]) : max(input_grid_2d[:, 1]) : gridsize * 1j,
    ]

    axs[0].set_title("Data")
    grid_data = griddata(
        points=input_grid_2d,
        values=targets,
        xi=(grid_x, grid_y),
        method="linear",
    )
    mappable_0 = axs[0].contourf(grid_x, grid_y, grid_data, levels=5, cmap="viridis")
    fig.colorbar(mappable_0, ax=axs[0])

    axs[1].set_title("CCE")
    grid_cce_means = griddata(input_grid_2d, cce_means, (grid_x, grid_y), method="linear")
    mappable_1 = axs[1].contourf(grid_x, grid_y, grid_cce_means, levels=5, cmap="viridis")
    fig.colorbar(mappable_1, ax=axs[1])

    axs[2].set_title("NLL")
    grid_nlls = griddata(input_grid_2d, nll_vals, (grid_x, grid_y), method="linear")
    mappable_2 = axs[2].contourf(grid_x, grid_y, grid_nlls, levels=5, cmap="viridis")
    fig.colorbar(mappable_2, ax=axs[2])

    fig.tight_layout()

    if show:
        plt.show()
    return fig


def main():
    dm = AAFDataModule(root_dir="data/aaf", batch_size=1, num_workers=0, persistent_workers=False)
    dm.setup("test")
    model = GaussianNN.load_from_checkpoint("chkp/aaf/gaussian/best_loss.ckpt", map_location="cpu")

    nlls = []
    test_targets = []
    with torch.inference_mode():
        for (x, y) in tqdm(dm.test_dataloader(), desc="Test inference"):
            y_hat = model.predict(x)
            nlls.append(-model.predictive_dist(y_hat).log_prob(y))
            test_targets.append(y)

    nlls = torch.cat(nlls)
    test_targets = torch.cat(test_targets)

    results = CalibrationResults.load("results/aaf/aaf_gaussian/calibration_results.pt")
    comp_fig = plot_cce_vs_nll(
        results.input_grid_2d, test_targets, results.cce.expected_values, nlls
    )
    comp_fig.savefig("probcal/figures/artifacts/aaf_cce_vs_nll.pdf", dpi=150)

    nll_hist_fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.histplot(nlls, stat="probability", bins=35, ax=ax)
    ax.set_xlabel("NLL")
    ax.set_ylabel("Density")
    nll_hist_fig.savefig("probcal/figures/artifacts/aaf_nll_hist.pdf", dpi=150)

    cce_hist_fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.histplot(results.cce.expected_values, stat="probability", bins=35, ax=ax)
    ax.set_xlabel("CCE")
    ax.set_ylabel("Density")
    cce_hist_fig.savefig("probcal/figures/artifacts/aaf_cce_hist.pdf", dpi=150)


if __name__ == "__main__":
    main()
