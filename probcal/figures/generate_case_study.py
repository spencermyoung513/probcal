from argparse import ArgumentParser
from pathlib import Path

from matplotlib import pyplot as plt

from probcal.custom_datasets import COCOPeopleDataset
from probcal.custom_datasets import RotatedMNIST
from probcal.custom_datasets.aaf_dataset import AAFDataset
from probcal.custom_datasets.readability_dataset import ReadabilityDataset
from probcal.enums import DatasetType
from probcal.enums import ImageDatasetName
from probcal.enums import TextDatasetName
from probcal.evaluation.calibration_evaluator import CalibrationResults


def generate_case_study_figures(dataset_type: DatasetType, dataset_name: str, head_alias: str):
    if dataset_type == DatasetType.IMAGE:
        dataset_name = ImageDatasetName(dataset_name)
        if dataset_name == ImageDatasetName.AAF:
            dataset = AAFDataset(root_dir="data/aaf", split="test")
        elif dataset_name == ImageDatasetName.COCO_PEOPLE:
            dataset = COCOPeopleDataset(root_dir="data/coco-people", split="test")
        elif dataset_name == ImageDatasetName.EVA:
            raise NotImplementedError()
        elif dataset_name == ImageDatasetName.FG_NET:
            raise NotImplementedError()
        elif dataset_name == ImageDatasetName.ROTATED_MNIST:
            dataset = RotatedMNIST(root_dir="data/rotated-mnist", split="test")
        else:
            raise NotImplementedError()
    elif dataset_type == DatasetType.TEXT:
        dataset_name = TextDatasetName(dataset_name)
        if dataset_name == TextDatasetName.READABILITY:
            dataset = ReadabilityDataset(root_dir="data/readability", split="test")
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError("Case studies are only produced for image/text data.")

    suffix = f"{dataset.root_dir.stem}/{head_alias}"
    results_dir = Path(f"results/{suffix}")
    save_dir = Path(f"probcal/figures/artifacts/case-studies/{suffix}")
    print(f"Saving case study figures to {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    results = CalibrationResults.load(results_dir / "calibration_results.pt")
    cce_means = results.cce.expected_values
    cce_stdevs = results.cce.stdevs

    # fig = CalibrationEvaluator.plot_cce_results(results, show=False)
    # fig.savefig(save_dir / "cce_grid.pdf", dpi=150)

    order = cce_means.argsort()

    # Plot examples with the lowest CCE.
    for i in range(5):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax: plt.Axes
        idx = order[i]
        idx = idx.item() if hasattr(idx, "item") else idx
        mean_cce = cce_means[idx]
        std_cce = cce_stdevs[idx]
        example, _ = dataset[idx]
        if dataset_type == DatasetType.IMAGE:
            ax.imshow(example)
            ax.set_title(f"{mean_cce:.4f} ({std_cce:.4f})")
        elif dataset_type == DatasetType.TEXT:
            pass
        ax.axis("off")
        fig.savefig(save_dir / f"cce_best_{i}.jpg", dpi=150)
        print(f"Saved {save_dir / f'cce_best_{i}.jpg'}")

    # Plot examples with the highest CCE.
    for i in range(5):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax: plt.Axes
        idx = order[-(i + 1)]
        idx = idx.item() if hasattr(idx, "item") else idx
        mean_cce = cce_means[idx]
        std_cce = cce_stdevs[idx]
        example, _ = dataset[idx]
        if dataset_type == DatasetType.IMAGE:
            ax.imshow(example)
            ax.set_title(f"{mean_cce:.4f} ({std_cce:.4f})")
        elif dataset_type == DatasetType.TEXT:
            pass
        ax.axis("off")
        fig.savefig(save_dir / f"cce_worst_{i}.jpg", dpi=150)
        print(f"Saved {save_dir / f'cce_worst_{i}.jpg'}")

        # Plot examples with the lowest CCE.
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    # Plot examples with the lowest CCE.
    for i in range(5):
        ax = axes[i]
        idx = order[i]
        idx = idx.item() if hasattr(idx, "item") else idx
        mean_cce = cce_means[idx]
        std_cce = cce_stdevs[idx]
        example, _ = dataset[idx]
        if dataset_type == DatasetType.IMAGE:
            ax.imshow(example)
            ax.set_title(f"{mean_cce:.4f} ({std_cce:.4f})")
        elif dataset_type == DatasetType.TEXT:
            pass
        ax.axis("off")

    # Plot examples with the highest CCE.
    for i in range(5):
        ax = axes[i + 5]
        idx = order[-(i + 1)]
        idx = idx.item() if hasattr(idx, "item") else idx
        mean_cce = cce_means[idx]
        std_cce = cce_stdevs[idx]
        example, _ = dataset[idx]
        if dataset_type == DatasetType.IMAGE:
            ax.imshow(example)
            ax.set_title(f"{mean_cce:.4f} ({std_cce:.4f})")
        elif dataset_type == DatasetType.TEXT:
            pass
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_dir / "cce_grid.jpg", dpi=150)
    print(f"Saved {save_dir / 'cce_grid.jpg'}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-type", type=DatasetType)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument(
        "--head",
        type=str,
        choices=["gaussian", "seitzer", "immer", "stirn", "poisson", "nbinom", "ddpn"],
    )
    args = parser.parse_args()
    generate_case_study_figures(
        dataset_type=args.dataset_type,
        dataset_name=str(args.dataset_name).replace("-", "_"),
        head_alias=args.head,
    )
