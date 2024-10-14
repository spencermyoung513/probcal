from functools import partial
from typing import Literal
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from seaborn import color_palette
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel

from probcal.evaluation.metrics import compute_mcmd_numpy


def generate_homoscedastic_data(
    n: int = 1000,
    mean_modification: Literal["default", "higher", "lower"] = "default",
    var_modification: Literal["default", "higher", "lower"] = "default",
) -> tuple[np.ndarray, np.ndarray]:
    """Create a homoscedastic x-conditional dataset.

    Args:
        n (int, optional): The number of samples to draw. Defaults to 1000.
        mean_modification (str, optional): Specifies how to modify the mean when generating ("default", "higher", "lower"). Defaults to "default".
        var_modification (str, optional): Specifies how to modify the variance when generating ("default", "higher", "lower"). Defaults to "default".

    Returns:
        tuple[np.ndarray, np.ndarray]: The resultant dataset (returned in x, y order).
    """
    x = np.random.uniform(0, 2 * np.pi, size=n)

    if mean_modification == "default":
        mean = np.cos(x)
    elif mean_modification == "higher":
        mean = np.cos(x) + 2
    elif mean_modification == "lower":
        mean = np.cos(x) - 2

    if var_modification == "default":
        stdev = 0.5
    elif var_modification == "higher":
        stdev = 1.5
    elif var_modification == "lower":
        stdev = 0.1

    y = np.random.normal(loc=mean, scale=stdev)

    return x, y


def generate_lambda_fig(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_prime: np.ndarray,
    y_prime: np.ndarray,
    gamma_x: np.ndarray,
    gamma_y: np.ndarray,
    lmbda_vals: np.ndarray,
):
    fig, axs = plt.subplots(2, 1, figsize=(5, 6), sharex="col")
    axs: Sequence[plt.Axes]
    colors = color_palette("rocket", n_colors=len(lmbda_vals))

    for lmbda, color in zip(lmbda_vals, colors):
        mcmd_vals = compute_mcmd_numpy(
            grid=grid,
            x=x,
            y=y,
            x_prime=x_prime,
            y_prime=y_prime,
            x_kernel=partial(rbf_kernel, gamma=gamma_x),
            y_kernel=partial(rbf_kernel, gamma=gamma_y),
            lmbda=lmbda,
        )
        order = np.argsort(grid)

        axs[1].plot(
            grid[order], mcmd_vals[order], label=f"$\lambda = {lmbda:.3f}$", color=color, alpha=0.8
        )

    axs[0].scatter(x, y, alpha=0.6)
    axs[0].set_ylabel("$Y$")
    axs[0].scatter(x_prime, y_prime, alpha=0.6)
    axs[1].set_xlabel("$X$")
    axs[1].set_ylabel("MCMD")

    # Move the legend under the bottom subplot and spread it out horizontally
    axs[1].legend(bbox_to_anchor=(0.5, -0.4), loc="center", ncol=len(lmbda_vals) // 2)
    plt.subplots_adjust(bottom=0.15)

    fig.tight_layout()  # Adjust layout to fit everything
    fig.savefig("probcal/figures/artifacts/lambda_effect.pdf", dpi=150)


def generate_gamma_fig(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_prime: np.ndarray,
    y_prime: np.ndarray,
):
    lmbda = 0.1
    gamma_vals = [0.1, 0.5, 1, 5]

    fig, axs = plt.subplots(2, len(gamma_vals), figsize=(8, 5), sharey="row")
    for i, gamma in enumerate(gamma_vals):

        sigma = np.power(2 * gamma, -0.5)

        axs[0, i].set_title(f"$\gamma = {gamma}$")
        axs[0, i].scatter(x, y, alpha=0.7)
        axs[0, i].scatter(x_prime, y_prime, alpha=0.7)

        for point in zip(x, y):
            ellipse = Ellipse(
                xy=point,
                width=2 * sigma,
                height=2 * sigma,
                edgecolor="red",
                fc="None",
                lw=1,
                alpha=0.1,
            )
            axs[0, i].add_patch(ellipse)
        for point in zip(x_prime, y_prime):
            ellipse = Ellipse(
                xy=point,
                width=2 * sigma,
                height=2 * sigma,
                edgecolor="red",
                fc="None",
                lw=1,
                alpha=0.1,
            )
            axs[0, i].add_patch(ellipse)

        mcmd_vals = compute_mcmd_numpy(
            grid=grid,
            x=x,
            y=y,
            x_prime=x_prime,
            y_prime=y_prime,
            x_kernel=partial(rbf_kernel, gamma=gamma),
            y_kernel=partial(rbf_kernel, gamma=gamma),
            lmbda=lmbda,
        )
        order = np.argsort(x)

        axs[1, 0].set_ylabel("MCMD")
        axs[1, i].plot(x[order], mcmd_vals[order])

    fig.tight_layout()
    fig.savefig("probcal/figures/artifacts/gamma_effect.pdf", dpi=150)


def generate_kernel_fig(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_prime: np.ndarray,
    y_prime: np.ndarray,
):
    lmbda = 0.1
    gamma_x = 2
    gamma_y = 1 / (2 * np.var(y))

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax: plt.Axes

    order = np.argsort(x)
    rbf_mcmd_vals = compute_mcmd_numpy(
        grid=grid,
        x=x,
        y=y,
        x_prime=x_prime,
        y_prime=y_prime,
        x_kernel=partial(rbf_kernel, gamma=gamma_x),
        y_kernel=partial(rbf_kernel, gamma=gamma_y),
        lmbda=lmbda,
    )
    laplacian_mcmd_vals = compute_mcmd_numpy(
        grid=grid,
        x=x,
        y=y,
        x_prime=x_prime,
        y_prime=y_prime,
        x_kernel=partial(laplacian_kernel, gamma=gamma_x),
        y_kernel=partial(laplacian_kernel, gamma=gamma_y),
        lmbda=lmbda,
    )
    polynomial_mcmd_vals = compute_mcmd_numpy(
        grid=grid,
        x=x,
        y=y,
        x_prime=x_prime,
        y_prime=y_prime,
        x_kernel=partial(polynomial_kernel, gamma=0.02),
        y_kernel=partial(polynomial_kernel, gamma=0.02),
        lmbda=lmbda,
    )

    order = np.argsort(grid)
    ax.set_xlabel("$X$")
    ax.set_ylabel("MCMD")
    ax.plot(grid[order], rbf_mcmd_vals[order], label="RBF")
    ax.plot(grid[order], laplacian_mcmd_vals[order], label="Laplacian")
    ax.plot(grid[order], polynomial_mcmd_vals[order], label="Polynomial")
    ax.legend()

    fig.tight_layout()
    fig.savefig("probcal/figures/artifacts/kernel_effect.pdf", dpi=150)


def main():
    x, y = generate_homoscedastic_data(n=200)
    x_prime = np.random.uniform(x.min(), x.max(), size=100)
    y_prime = np.random.normal(x_prime - 5, 0.5)

    grid = np.linspace(0, 2 * np.pi, num=100)
    gamma_x = 0.5
    gamma_y = 1 / (2 * np.var(y, ddof=1))
    lmbda_vals = [0.001, 0.01, 0.05, 0.1, 0.5, 1]

    generate_lambda_fig(grid, x, y, x_prime, y_prime, gamma_x, gamma_y, lmbda_vals)
    generate_gamma_fig(grid, x, y, x_prime, y_prime)
    generate_kernel_fig(grid, x, y, x_prime, y_prime)
