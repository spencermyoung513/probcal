import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import Resize


def main(model_type: str):
    save_dir = Path(f"probcal/figures/artifacts/coco-people-diagnostics/{model_type}")
    if not save_dir.exists():
        os.makedirs(save_dir)

    path_to_results = f"probcal/figures/artifacts/coco-people-cce-vals/{model_type}_results.json"

    with open(path_to_results) as f:
        cce_results: dict = json.load(f)

    cce_values = np.array([float(x) for x in cce_results.values()])
    image_paths = list(cce_results.keys())
    cce_low_to_high = np.argsort(cce_values)

    n = 5
    least_aligned = cce_low_to_high[-n:]
    most_aligned = cce_low_to_high[:n]
    resize = Resize((224, 224))

    fig, axs = plt.subplots(2, n, figsize=(n * 2, 6))
    for i in range(n):
        axs[0, i].set_title(f"CCE: {cce_values[most_aligned[i]]:.4f}", fontsize=10)
        axs[0, i].imshow(resize(Image.open(image_paths[most_aligned[i]])))
        axs[1, i].set_title(f"CCE: {cce_values[least_aligned[i]]:.4f}", fontsize=10)
        axs[1, i].imshow(resize(Image.open(image_paths[least_aligned[i]])))

    fig.text(0.1, 0.7, "Lowest CCE", va="center", ha="center", rotation="vertical", fontsize=10)
    fig.text(0.1, 0.3, "Highest CCE", va="center", ha="center", rotation="vertical", fontsize=10)
    [ax.axis("off") for ax in axs.ravel()]
    fig.savefig(save_dir / "cce_best_worst.pdf", dpi=150)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["ddpn", "faithful", "gaussian", "natural", "nbinom", "poisson", "seitzer"],
    )
    args = parser.parse_args()
    main(args.model_type)
