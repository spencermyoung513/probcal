import torch
import numpy as np
from functools import partial

import argparse
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel

from probcal.utils.experiment_utils import get_model, get_datamodule
from probcal.utils.configs import TrainingConfig

from probcal.samplers import SAMPLERS
from probcal.evaluation.metrics import compute_mcmd
from probcal.evaluation.plotting import plot_hex_bin_mcmd

from probcal.enums import DatasetType, ImageDatasetName, HeadType

MODELS = [member.value for member in HeadType]

def init_y_hat_array(model: str, n: int) -> torch.Tensor:
    if model == "gaussian" or model == "nbinom":
        y_hat_init = torch.zeros((n, 2))
    else:
        y_hat_init = torch.zeros((n, 1))
    return y_hat_init


def main(model_type: str, num_samples: int):
    print(f"Executing MNIST Experiment: {model_type}")
    datamodule = get_datamodule(
        DatasetType.IMAGE,
        ImageDatasetName.MNIST,
        1
    )
    datamodule.setup("predict")
    test_loader = datamodule.test_dataloader()

    cfg = TrainingConfig.from_yaml(f"../../configs/mnist_{model_type}_cfg.yaml")
    model = get_model(cfg)

    state_dict = torch.load(
        f"/Users/porterjenkins/code/probcal/weights/mnist_{model_type}/best_mae_{model_type}.pt",
        map_location='cpu'
    )
    model.load_state_dict(state_dict)

    num_examples = 1000
    X = np.zeros((num_examples, 28*28))
    Y = np.zeros((num_examples, 1))
    Y_hat = init_y_hat_array(model_type, num_examples)

    for i, (x, y) in enumerate(test_loader):
        if i == num_examples:
            break
        X[i] = x.flatten()
        Y[i] = y
        with torch.no_grad():
            y_hat = model._predict_impl(x).squeeze()
        Y_hat[i] = y_hat

    sampler = SAMPLERS[model_type](Y_hat)

    y_prime = sampler.sample(m=num_samples).reshape(-1, 1)
    print(np.mean(y_prime), np.std(y_prime))
    x_kernel = partial(rbf_kernel, gamma=5.0)
    y_kernel = partial(rbf_kernel, gamma=0.5)
    print("Computing TSNE")
    X_reduced = TSNE(n_components=2, random_state=1990).fit_transform(X)

    print('Computing MCMD')
    mcmd_vals = compute_mcmd(
                grid=X_reduced,
                x=X_reduced,
                y=Y,
                x_prime=np.tile(X_reduced, (num_samples, 1)),
                y_prime=y_prime,
                x_kernel=x_kernel,
                y_kernel=y_kernel,
            )

    print(mcmd_vals)
    mean_mcmd = np.mean(mcmd_vals)
    print(mean_mcmd)
    total_mcmd = np.sum(mcmd_vals)
    print(total_mcmd)

    plot_hex_bin_mcmd(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        c=mcmd_vals,
        title="MCMD on MNIST: {} DNN ({:.3f})".format(model_type, mean_mcmd),
        grid_size=20,
        fpath=f"../figures/artifacts/mnist_{model_type}_mcmd.png"
    )

if __name__ == "__main__":
    # TODO: consider adding experiment config instead of cli args
    parser = argparse.ArgumentParser(description="Run the MCMD experiment on MNIST")
    parser.add_argument("--model", type=str, choices=MODELS, help=f"must be in {MODELS}")
    parser.add_argument("--num-samples", type=int, default=10, help=f"number of predictive draws")
    args = parser.parse_args()
    main(
        model_type=args.model,
        num_samples=args.num_samples
    )