import json
import os
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image


def main(model_type: str):
    save_dir = Path(f"probcal/figures/artifacts/coco-people-diagnostics/{model_type}")
    if not save_dir.exists():
        os.makedirs(save_dir)

    path_to_results = f"probcal/figures/artifacts/coco-people-cce-vals/{model_type}_results.json"

    with open(path_to_results) as f:
        results: dict = json.load(f)
    worst_5 = list(reversed(sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]))
    best_5 = sorted(results.items(), key=lambda x: x[1])[:5]

    for i, (path, cce) in enumerate(worst_5):
        img = Image.open(path).resize((224, 224))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img.save(f"{save_dir}/cce_worst_{i}_{cce:.4f}.jpg")

    for i, (path, cce) in enumerate(best_5):
        img = Image.open(path).resize((224, 224))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img.save(f"{save_dir}/cce_best_{i}_{cce:.4f}.jpg")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["faithful", "gaussian", "natural", "nbinom", "poisson", "seitzer"],
    )
    args = parser.parse_args()
    main(args.model_type)
