from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import TypeAlias

import lightning as L
import open_clip
import torch.nn.functional
from matplotlib import pyplot as plt
from open_clip import CLIP
from scipy.interpolate import griddata
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from probcal.enums import DatasetType
from probcal.evaluation.kernels import bhattacharyya_kernel
from probcal.evaluation.kernels import laplacian_kernel
from probcal.evaluation.kernels import polynomial_kernel
from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.evaluation.metrics import compute_regression_ece
from probcal.models.regression_nn import RegressionNN


@dataclass
class CCEResult:
    cce_vals: np.ndarray
    mean_cce: float


@dataclass
class ProbabilisticResults:
    input_grid_2d: np.ndarray
    regression_targets: np.ndarray
    cce_results: list[CCEResult]
    ece: float

    def save(self, filepath: str | Path):
        if not str(filepath).endswith(".npz"):
            raise ValueError("Filepath must have a .npz extension.")
        save_dict = {
            "input_grid_2d": self.input_grid_2d,
            "regression_targets": self.regression_targets,
            "ece": self.ece,
        }
        save_dict.update(
            {f"cce_vals_{i}": self.cce_results[i].cce_vals for i in range(len(self.cce_results))}
        )
        save_dict.update(
            {f"mean_cce_{i}": self.cce_results[i].mean_cce for i in range(len(self.cce_results))}
        )
        np.savez(filepath, **save_dict)

    @staticmethod
    def load(filepath: str | Path) -> ProbabilisticResults:
        data: dict[str, np.ndarray] = np.load(filepath)
        num_trials = max(
            int(k.split("mean_cce_")[-1]) + 1 for k in data.keys() if k.startswith("mean_cce_")
        )
        cce_results = [
            CCEResult(
                cce_vals=data[f"cce_vals_{i}"],
                mean_cce=data[f"mean_cce_{i}"],
            )
            for i in range(num_trials)
        ]
        return ProbabilisticResults(
            input_grid_2d=data["input_grid_2d"],
            regression_targets=data["regression_targets"],
            cce_results=cce_results,
            ece=data["ece"].item(),
        )


KernelFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class ProbabilisticEvaluatorSettings:
    dataset_type: DatasetType = DatasetType.IMAGE
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cce_use_val_split_for_S: bool = False
    cce_num_trials: int = 5
    cce_input_kernel: Literal["polynomial"] | KernelFunction = "polynomial"
    cce_output_kernel: Literal["rbf", "laplacian", "bhatt"] | KernelFunction = "rbf"
    cce_lambda: float = 0.1
    cce_num_samples: int = 1
    ece_bins: int = 50
    ece_weights: Literal["uniform", "frequency"] = "frequency"
    ece_alpha: float = 1.0


class ProbabilisticEvaluator:
    """Helper object to evaluate the probabilistic fit of a neural net."""

    def __init__(self, settings: ProbabilisticEvaluatorSettings):
        self.settings = settings
        self.device = settings.device
        self._clip_model = None
        self._image_preprocess = None
        self._tokenizer = None

    @torch.inference_mode()
    def __call__(
        self, model: RegressionNN, data_module: L.LightningDataModule
    ) -> ProbabilisticResults:
        model.to(self.device)
        data_module.prepare_data()
        data_module.setup("test")
        val_dataloader = data_module.val_dataloader()
        test_dataloader = data_module.test_dataloader()
        # file_to_cce = {}

        print(f"Running {self.settings.cce_num_trials} CCE computation(s)...")
        cce_results = []

        # for saving label images with cce values
        output_dir = Path("cce_images")
        output_dir.mkdir(exist_ok=True)

        for i in range(self.settings.cce_num_trials):
            cce_vals, grid, targets, images = self.compute_cce(
                model=model,
                grid_loader=test_dataloader,
                sample_loader=val_dataloader,
                test_loader=test_dataloader
                if self.settings.cce_use_val_split_for_S
                else test_dataloader,
                return_grid=True,
                return_targets=True,
            )

            cce_vals_np = cce_vals.detach().cpu().numpy()
            images = images.detach().cpu()

            assert (
                cce_vals_np.shape[0] == images.shape[0]
            ), "CCE values and images are not aligned!"

            # Get indices of the top 5 highest CCE values
            top5_indices = np.argsort(cce_vals_np)[-5:]  # Highest CCE values

            # Get indices of the bottom 5 lowest CCE values
            bottom5_indices = np.argsort(cce_vals_np)[:5]  # Lowest CCE values

            # Combine the indices
            selected_indices = np.concatenate([bottom5_indices, top5_indices])

            # Loop over selected indices to process and save images
            for idx in selected_indices:
                image = images[idx]  # Get the image at the index
                cce_value = cce_vals_np[idx]

                # Denormalize the image
                mean = 0.1307
                std = 0.3081
                image_denorm = image * std + mean
                image_denorm = image_denorm.clamp(0, 1)

                # Convert image tensor to NumPy array
                image_np = image_denorm.squeeze().numpy()  # Remove channel dimension if needed

                # Plot the image
                plt.figure()
                plt.imshow(image_np, cmap="gray")
                plt.title(f"CCE: {cce_value:.4f}")
                plt.axis("off")

                # Save the image with CCE value in the filename
                filename = output_dir / f"cce_{cce_value:.4f}_idx_{idx}.png"
                plt.savefig(filename)
                plt.close()

                print(f"Image saved: {filename}")

            # We only need to save the input grid / regression targets once.
            if i == 0:
                if self.settings.dataset_type == DatasetType.TABULAR:
                    grid_2d = np.array([])
                else:
                    print("Running TSNE to project grid to 2d...")
                    grid_2d = TSNE().fit_transform(grid.detach().cpu().numpy())
                regression_targets = targets.detach().cpu().numpy()

            cce_results.append(
                CCEResult(
                    cce_vals=cce_vals.detach().cpu().numpy(),
                    mean_cce=cce_vals.mean().item(),
                )
            )

        print("Computing ECE...")
        ece = evaluate_model_calibration(model, test_dataloader)
        print("ece", ece)

        print("input grid 2d", grid_2d)
        print("cce results", cce_results)

        return ProbabilisticResults(
            input_grid_2d=grid_2d,
            regression_targets=regression_targets,
            cce_results=cce_results,
            ece=ece,
        )

    def compute_cce(
        self,
        model: RegressionNN,
        grid_loader: DataLoader,
        sample_loader: DataLoader,
        test_loader: DataLoader,
        return_grid: bool = False,
        return_targets: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Compute the CCE between the given model and data samples.

        Args:
            model (DiscreteRegressionNN): Probabilistic regression model to compute the CCE for.
            grid_loader (DataLoader): DataLoader with the data inputs to compute CCE over.
            sample_loader (DataLoader): DataLoader with the data inputs/outputs that define S (from which we form our two samples).
            return_grid (bool, optional): Whether/not to return the grid of values the CCE was computed over. Defaults to False.
            return_targets (bool, optional): Whether/not to return the regression targets the CCE was computed against. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The computed CCE values, along with the grid of inputs these values correspond to (if return_grid is True) and the regression targets (if return_targets is True).
        """
        x, y, x_prime, y_prime = self._get_samples_for_mcmd(model, sample_loader)
        # Converting using Torch for Greyscale to RGB
        # grid = torch.cat(
        #     [
        #         self.clip_model.encode_image(
        #             F.resize(inputs.repeat(1, 3, 1, 1), size=[224, 244], antialias=True).to(
        #                 self.device
        #             ),
        #             normalize=False,
        #         )
        #         for inputs, _ in grid_loader
        #     ],
        #     dim=0,
        # )
        # Converting using TSNE

        grid = torch.cat(
            [
                torch.Tensor(
                    TSNE(n_components=3, random_state=1990, perplexity=5).fit_transform(
                        inputs.reshape(inputs.shape[0], -1).numpy()
                    )
                )
                for inputs, _ in grid_loader
            ],
        )

        x_kernel, y_kernel = self._get_kernel_functions(y)
        print("Computing CCE...")
        print(grid.shape)
        print(x.shape)
        print(x_prime.shape)
        cce_vals = compute_mcmd_torch(
            grid=grid,
            x=x,
            y=y,
            x_prime=x_prime,
            y_prime=y_prime,
            x_kernel=x_kernel,
            y_kernel=y_kernel,
            lmbda=self.settings.cce_lambda,
        )
        # TODO: create a graph of some cce_vals from images in the grid_loader
        # # -> concat all the images similar to the grid thing above
        # # something like this
        print("getting tensor grid of test image greyscale values")
        images = torch.cat([inputs for inputs, _ in grid_loader], dim=0)
        # # -> maybe find the top 10 lowest/highest CCE vals
        return_obj = [cce_vals]
        if return_grid:
            return_obj.append(grid)
        if return_targets:
            return_obj.append(y)
        if images is not None:
            return_obj.append(images)
        if len(return_obj) == 1:
            return return_obj[0]
        else:
            return tuple(return_obj)

    def compute_ece(self, model: RegressionNN, data_loader: DataLoader) -> float:
        """Compute the regression ECE of the given model over the dataset spanned by the data loader.

        Args:
            model (DiscreteRegressionNN): Probabilistic regression model to compute the ECE for.
            data_loader (DataLoader): DataLoader with the test data to compute ECE over.

        Returns:
            float: The regression ECE.
        """
        all_outputs = []
        all_targets = []
        for inputs, targets in tqdm(
            data_loader, desc="Getting posterior predictive dists for ECE..."
        ):
            all_outputs.append(model.predict(inputs.to(self.device)))
            all_targets.append(targets.to(self.device))

        all_targets = torch.cat(all_targets).detach().cpu().numpy()
        all_outputs = torch.cat(all_outputs, dim=0)
        posterior_predictive = model.posterior_predictive(all_outputs)

        ece = compute_regression_ece(
            y_true=all_targets,
            posterior_predictive=posterior_predictive,
            num_bins=self.settings.ece_bins,
            weights=self.settings.ece_weights,
            alpha=self.settings.ece_alpha,
        )
        return ece

    def plot_cce_results(
        self,
        results: ProbabilisticResults,
        gridsize: int = 100,
        trial_index: int = 0,
        show: bool = False,
    ) -> plt.Figure:
        """Given a set of evaluation results and an existing axes, plot the CCE values against their 2d input projections on a hexbin grid.

        Args:
            results (EvaluationResults): Results from a ProbabilsticEvaluator.
            ax (plt.Axes): The axes to draw the plot on.
            gridsize (int, optional): Determines the granularity of the contour plot (higher numbers are more granular). Defaults to 100.
            trial_index (int, optional): Which CCE computation to use for the plot (since multiple trials are possible). Defaults to 0 (the first).
            show (bool, optional): Whether/not to show the resultant figure with plt.show(). Defaults to False.

        Returns:
            Figure: Matplotlib figure with the visualized CCE values.
        """
        cce_vals = results.cce_results[trial_index].cce_vals
        mean_cce = results.cce_results[trial_index].mean_cce
        input_grid_2d = results.input_grid_2d

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs: Sequence[plt.Axes]
        grid_x, grid_y = np.mgrid[
            min(input_grid_2d[:, 0]) : max(input_grid_2d[:, 0]) : gridsize * 1j,
            min(input_grid_2d[:, 1]) : max(input_grid_2d[:, 1]) : gridsize * 1j,
        ]

        axs[0].set_title("Data")
        grid_data = griddata(
            input_grid_2d,
            results.regression_targets,
            (grid_x, grid_y),
            method="linear",
        )
        mappable_0 = axs[0].contourf(grid_x, grid_y, grid_data, levels=5, cmap="viridis")
        fig.colorbar(mappable_0, ax=axs[0])

        axs[1].set_title(f"Mean CCE: {mean_cce:.4f}")
        grid_cce = griddata(input_grid_2d, cce_vals, (grid_x, grid_y), method="linear")
        mappable_1 = axs[1].contourf(grid_x, grid_y, grid_cce, levels=5, cmap="viridis")
        fig.colorbar(mappable_1, ax=axs[1])

        fig.tight_layout()

        if show:
            plt.show()
        return fig

    def _get_samples_for_mcmd(
        self, model: RegressionNN, data_loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = []
        y = []
        x_prime = []
        y_prime = []
        for inputs, targets in tqdm(
            data_loader, desc="Sampling from posteriors for MCMD computation..."
        ):
            if self.settings.dataset_type == DatasetType.TABULAR:
                x.append(inputs)
            elif self.settings.dataset_type == DatasetType.IMAGE:
                flattened = inputs.reshape(inputs.shape[0], -1)
                x.append(
                    torch.Tensor(
                        TSNE(n_components=3, random_state=1990, perplexity=5).fit_transform(
                            flattened.numpy()
                        )
                    )
                )
            elif self.settings.dataset_type == DatasetType.TEXT:
                x.append(self.clip_model.encode_text(inputs.to(self.device), normalize=False))
            y.append(targets.to(self.device))
            y_hat = model.predict(inputs.to(self.device))
            x_prime.append(
                torch.repeat_interleave(x[-1], repeats=self.settings.cce_num_samples, dim=0)
            )
            y_prime.append(apply_softmax(y_hat))

        x = torch.cat(x, dim=0)
        x = (x - x.mean(dim=0)) / x.std(dim=0)
        y = torch.cat(y).float()
        y = one_hot_encode_mnist(y)
        x_prime = torch.cat(x_prime, dim=0)
        x_prime = (x_prime - x_prime.mean(dim=0)) / x_prime.std(dim=0)
        y_prime = torch.cat(y_prime).float()

        return x, y, x_prime, y_prime

    def _get_kernel_functions(self, y: torch.Tensor) -> tuple[KernelFunction, KernelFunction]:
        if self.settings.cce_input_kernel == "polynomial":
            x_kernel = polynomial_kernel
        else:
            x_kernel = self.settings.cce_input_kernel

        if self.settings.cce_output_kernel == "rbf":
            y_kernel = partial(rbf_kernel, gamma=(1 / (2 * y.float().var())).item())
        elif self.settings.cce_output_kernel == "laplacian":
            y_kernel = partial(laplacian_kernel, gamma=(1 / (2 * y.float().var())).item())
        elif self.settings.cce_output_kernel == "bhatt":
            y_kernel = bhattacharyya_kernel

        return x_kernel, y_kernel

    @property
    def clip_model(self) -> CLIP:
        if self._clip_model is None:
            self._clip_model, _, self._image_preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                device=self.device,
            )
        return self._clip_model

    @property
    def image_preprocess(self) -> Compose:
        if self._image_preprocess is None:
            self._clip_model, _, self._image_preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                device=self.device,
            )
        return self._image_preprocess

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return self._tokenizer


def one_hot_encode_mnist(labels, num_classes=10):
    """
    One-hot encodes MNIST labels.

    Args:
        labels (torch.Tensor): Tensor of class labels (integers 0-9)
        num_classes (int): Number of classes (default=10 for MNIST)

    Returns:
        torch.Tensor: One-hot encoded labels
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Ensure labels are long/integer type
    labels = labels.long()

    # Handle both single labels and batches
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    # Convert to one-hot encoding
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)

    # Convert to float for compatibility with most loss functions
    return one_hot.float()


def apply_softmax(predictions, dim=1, temperature=1.0):
    """
    Applies softmax to model predictions with optional temperature scaling.

    Args:
        predictions (torch.Tensor): Raw model outputs/logits
        dim (int): Dimension along which to apply softmax (default=1 for batched predictions)
        temperature (float): Temperature for scaling predictions (default=1.0)
                           Higher values make distribution more uniform
                           Lower values make it more peaked

    Returns:
        torch.Tensor: Softmax probabilities
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)

    # Apply temperature scaling
    scaled_predictions = predictions / temperature

    # Handle both single predictions and batches
    if predictions.dim() == 1:
        # For single prediction, use dim=0
        return torch.nn.functional.softmax(scaled_predictions, dim=0)
    else:
        # For batched predictions, use specified dim (default=1)
        return torch.nn.functional.softmax(scaled_predictions, dim=dim)


import torch
import torch.nn.functional as F
import numpy as np


def compute_calibration_error(model, data_loader, n_bins=10, device="cpu"):
    """
    Computes the Expected Calibration Error (ECE) for a trained MNIST model.

    Args:
        model: The trained neural network
        data_loader: DataLoader containing MNIST test data
        n_bins: Number of confidence bins
        device: Device to run computation on

    Returns:
        ece: Expected Calibration Error
        confidences: List of confidence values for plotting
        accuracies: List of accuracy values for plotting
    """
    model.eval()
    confidences = []
    predictions = []
    labels = []

    # Collect model predictions and confidences
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Get model outputs
            logits = model(images)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=1)

            # Get confidence (maximum probability)
            conf, pred = torch.max(probs, dim=1)

            confidences.extend(conf.cpu().numpy())
            predictions.extend(pred.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    confidences = np.array(confidences)
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Create confidence bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries) - 1

    # Initialize arrays for storing bin statistics
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Compute statistics for each bin
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.any(mask):
            bin_accuracies[bin_idx] = np.mean(predictions[mask] == labels[mask])
            bin_confidences[bin_idx] = np.mean(confidences[mask])
            bin_counts[bin_idx] = np.sum(mask)

    # Compute ECE
    ece = np.sum((bin_counts / len(predictions)) * np.abs(bin_accuracies - bin_confidences))

    return ece, bin_confidences, bin_accuracies


# Example usage:
def evaluate_model_calibration(model, test_loader, device="cpu"):
    ece, confidences, accuracies = compute_calibration_error(
        model, test_loader, n_bins=10, device=device
    )
    print(f"Expected Calibration Error: {ece:.3f}")

    # Optional: Plot reliability diagram
    plot_reliability_diagram(ece, confidences, accuracies)

    return ece


def plot_reliability_diagram(ece, confidences, accuracies):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "r--")  # Perfect calibration line
    plt.plot(confidences, accuracies, "b.-", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram For ECE : {ece:.3f}")
    plt.legend()
    plt.grid(True)
    plt.savefig("cce_images/plots/reliability_diagram.png")
