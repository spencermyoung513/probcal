from typing import Optional

import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import rv_continuous


def plot_posterior_predictive(
    x: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    error_color: str = "r",
    error_alpha: float = 0.2,
    show: bool = True,
    legend: bool = True,
    title: str = "",
    ax: plt.Axes | None = None,
    ylims: tuple[float] | None = None,
):
    """Visualize a model's posterior predictive distribution over a 1d dataset (`x`, `y` both scalars) by showing the expected value and error bounds across the regression targets.

    Args:
        x (np.ndarray): The x values (inputs).
        y (np.ndarray): The ground truth y values (outputs).
        mu (np.ndarray): The expected values of the model's posterior predictive distribution over `y`.
        upper (np.ndarray): Upper bounds for the model's posterior predictive distribution over `y`.
        lower (np.ndarray): Lower bounds for the model's posterior predictive distribution over `y`.
        error_color (str, optional): Color with which to fill the model's error bounds. Defaults to "r".
        alpha (float, optional): Transparency value for the model's error bounds. Defaults to 0.2.
        show (bool, optional): Whether/not to show the resultant plot. Defaults to True.
        legend (bool, optional): Whether/not to put a legend in the resultant plot. Defaults to True.
        title (str, optional): If specified, a title for the resultant plot. Defaults to "".
        ax (plt.Axes | None, optional): If given, the axis on which to plot the posterior predictive distribution. Defaults to None (axis is created).
        ylims (tuple[float] | None, optional): If given, the lower/upper axis limits for the plot. Defaults to None.
    """
    order = x.argsort()

    ax = plt.subplots(1, 1, figsize=(10, 6))[1] if ax is None else ax

    ax.scatter(x[order], y[order], alpha=0.1, label="Test Data")
    ax.plot(x[order], mu[order])
    ax.fill_between(
        x[order], lower[order], upper[order], color=error_color, alpha=error_alpha, label="95% CI"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if legend:
        ax.legend()
    ax.set_title(title)
    if ylims is None:
        ax.set_ylim(lower.min() - 5, upper.max() + 5)
    else:
        ax.set_ylim(*ylims)
    if show:
        plt.show()


def plot_regression_calibration_curve(
    y_true: np.ndarray,
    posterior_predictive: rv_continuous,
    num_bins: int = 9,
    ax: plt.Axes | None = None,
    show: bool = True,
):
    """Given targets and a probabilistic regression model (represented as a continuous random variable over the targets), plot a calibration curve.

    Args:
        y_true (ndarray, (n,)): The true values of the regression targets.
        posterior_predictive (rv_continuous): Random variable representing the posterior predictive distribution over the targets.
        num_bins (int): Specifies how many probability thresholds to use for checking CDF calibration. This
                        corresponds to how many points will be plotted to form the calibration curve.
        ax (plt.Axes | None): The axis to plot on (if provided). If None is passed in, an axis is created.
        show (bool): Specifies whether/not to display the resultant plot.
    """
    epsilon = 1e-4
    p_vals = np.linspace(0 + epsilon, 1 - epsilon, num=num_bins).reshape(-1, 1)
    expected_pct_where_cdf_less_than_p = p_vals
    actual_pct_where_cdf_less_than_p = (posterior_predictive.cdf(y_true) <= p_vals).mean(axis=1)

    ax = plt.subplots(1, 1)[1] if ax is None else ax
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Expected Confidence Level")
    ax.set_ylabel("Observed Confidence Level")
    ax.plot(
        expected_pct_where_cdf_less_than_p,
        expected_pct_where_cdf_less_than_p,
        linestyle="--",
        color="red",
        label="Perfectly calibrated",
    )
    ax.plot(
        expected_pct_where_cdf_less_than_p,
        actual_pct_where_cdf_less_than_p,
        marker="o",
        linestyle="-",
        color="black",
        label="Model",
    )
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()


def plot_hex_bin_mcmd(
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        grid_size: Optional[int] = 25,
        title: Optional[str] = "Hexbin plot",
        fpath: Optional[str] = None,
        crange: Optional[tuple[float, float]] = None
):
    """
        Generates a hexbin plot, which is a two-dimensional histogram with hexagonal bins, from three numpy arrays representing x and y coordinates and a value to aggregate.

        Args:
            x (np.ndarray): The x coordinates for each point.
            y (np.ndarray): The y coordinates for each point.
            c (np.ndarray): The values to aggregate over the hexagonal bins. The color of each hexagon is determined by the average value of `c` in that bin.
            grid_size (Optional[int], optional): The number of hexagons across the x-axis of the plot. Defaults to 25.
            title (Optional[str], optional): The title of the plot. Defaults to "Hexbin plot".
            fpath (Optional[str], optional): The file path to save the plot to. If None, the plot is displayed using plt.show(). Defaults to None.
            crange (Optional[tuple[float, float]], optional): The range of values to normalize the colors of the hexagons. Defaults to None, which automatically scales to the min and max of `c`.

        Returns:
            None: This function does not return a value but generates and displays or saves a hexbin plot.
        """

    # Create the hexbin plot
    plt.figure(figsize=(8, 6))
    if crange is not None:
        hb = plt.hexbin(x, y, C=c, gridsize=grid_size, reduce_C_function=np.mean, cmap='viridis', vmin=crange[0], vmax=crange[1])
    else:
        hb = plt.hexbin(x, y, C=c, gridsize=grid_size, reduce_C_function=np.mean, cmap='viridis')

    # Add a color bar
    cb = plt.colorbar(hb)
    cb.set_label('Average MCMD')

    # Add labels and title
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(title)

    if fpath is not None:
        plt.savefig(fpath)
    else:
        plt.show()

def get_scatter_plot_by_cls(x: np.ndarray, y: np.ndarray, c: np.ndarray, title:str, fpath: Optional[str] = None):

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=c, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    if fpath is not None:
        plt.savefig(fpath)
    else:
        plt.show()