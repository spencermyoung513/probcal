from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel

from probcal.evaluation.metrics import compute_mcmd_numpy
from probcal.models import DoublePoissonNN
from probcal.models import GaussianNN
from probcal.models import NegBinomNN
from probcal.models import PoissonNN
from probcal.models.regression_nn import RegressionNN


def produce_figure(
    models: list[RegressionNN],
    names: list[str],
    save_path: Path | str,
    dataset_path: Path | str,
):
    """Create a figure showcasing how CCE behaves with uniform downsampling.

    Args:
        models (list[DiscreteRegressionNN]): List of models to plot CCE distributions of.
        names (list[str]): List of display names for each respective model in `models`.
        save_path (Path | str): Path to save figure to.
        dataset_path (Path | str): Path with dataset models were fit on.
    """
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, axs = plt.subplots(1, 4, figsize=(6, 3), sharey="row")
    data: dict[str, np.ndarray] = np.load(dataset_path)
    X = data["X_test"].flatten()
    y = data["y_test"].flatten()
    x_kernel = partial(rbf_kernel, gamma=0.5)

    n = len(X)
    sample_sizes = [25, 50, 75, 100]
    colors = ["b", "g", "r", "c"]

    for i, sample_size in enumerate(sample_sizes):
        indices = np.random.choice(n, size=sample_size, replace=False)
        y_kernel = partial(rbf_kernel, gamma=1 / (2 * y[indices].var(ddof=1)))

        mean_cce_vals = {name: [] for name in names}
        for j, (model, model_name) in enumerate(zip(models, names)):
            for _ in range(5):

                y_hat = model.predict(torch.tensor(X[indices]).unsqueeze(1))
                y_prime = model.sample(y_hat, training=False).flatten().detach().numpy()
                cce_vals = compute_mcmd_numpy(
                    grid=X[indices],
                    x=X[indices],
                    y=y[indices],
                    x_prime=X[indices],
                    y_prime=y_prime,
                    x_kernel=x_kernel,
                    y_kernel=y_kernel,
                    lmbda=0.1,
                )
                mean_cce_vals[model_name].append(cce_vals.mean())

        means = [np.mean(x) for x in mean_cce_vals.values()]
        stdevs = [np.std(x, ddof=1) for x in mean_cce_vals.values()]
        for j in range(len(means)):
            axs[i].errorbar(
                j,
                means[j],
                yerr=stdevs[j],
                fmt="o",
                color=colors[j],
                capsize=4,
                markersize=5,
                label=names[j],
            )
        axs[i].set_title(f"$k = {sample_size}$")
        axs[i].set_xticks([])
        axs[i].set_ylim(-0.01, 0.3)
        axs[i].set_xlim(-0.5, 3.5)

    axs[0].set_ylabel(r"$\overline{\mathrm{CCE}}$")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.15), loc="upper center", ncol=4)
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(save_path, format="pdf", dpi=150)


if __name__ == "__main__":
    save_path = "probcal/figures/artifacts/cce_with_downsampling.pdf"
    dataset_path = "data/discrete_sine_wave/discrete_sine_wave.npz"
    models = [
        PoissonNN.load_from_checkpoint("weights/discrete_sine_wave_poisson.ckpt"),
        NegBinomNN.load_from_checkpoint("weights/discrete_sine_wave_nbinom.ckpt"),
        GaussianNN.load_from_checkpoint("weights/discrete_sine_wave_gaussian.ckpt"),
        DoublePoissonNN.load_from_checkpoint("weights/discrete_sine_wave_ddpn.ckpt"),
    ]
    names = ["Poisson NN", "NB NN", "Gaussian NN", "Double Poisson NN"]
    produce_figure(models, names, save_path, dataset_path)
