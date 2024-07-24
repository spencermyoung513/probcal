import torch
import numpy as np
from functools import partial
import yaml
import pprint

import argparse
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel

from probcal.utils.experiment_utils import get_model, get_datamodule
from probcal.utils.configs import TrainingConfig

from probcal.samplers import SAMPLERS
from probcal.evaluation.metrics import compute_mcmd
from probcal.evaluation.plotting import plot_hex_bin_mcmd, get_scatter_plot_by_cls

from probcal.enums import DatasetType, ImageDatasetName, HeadType


MODELS = [member.value for member in HeadType]

def init_y_hat_array(model: str, n: int) -> torch.Tensor:
    if model == "gaussian" or model == "nbinom":
        y_hat_init = torch.zeros((n, 2))
    else:
        y_hat_init = torch.zeros((n, 1))
    return y_hat_init


def main(cfg: dict):
    print(f"Executing MNIST Experiment:")
    pprint.pprint(cfg)

    datamodule = get_datamodule(
        DatasetType.IMAGE,
        ImageDatasetName.MNIST,
        1
    )
    datamodule.setup("predict")
    test_loader = datamodule.test_dataloader()

    model_cfg = TrainingConfig.from_yaml(f"configs/mnist_{cfg['model_type']}_cfg.yaml")
    model = get_model(model_cfg)

    state_dict = torch.load(
        f"weights/mnist_{cfg['model_type']}/best_mae_{cfg['model_type']}.pt",
        map_location='cpu'
    )
    model.load_state_dict(state_dict)

    num_examples = cfg['data']['test_examples']
    X = np.zeros((num_examples, 28*28))
    Y = np.zeros((num_examples, 1))
    Y_hat = init_y_hat_array(cfg['model_type'], num_examples)

    for i, (x, y) in enumerate(test_loader):
        if i == num_examples:
            break
        X[i] = x.flatten()
        Y[i] = y
        with torch.no_grad():
            y_hat = model._predict_impl(x).squeeze()
        Y_hat[i] = y_hat

    sampler = SAMPLERS[cfg['model_type']](Y_hat)

    y_prime = sampler.sample(m=cfg['data']['n_draws']).reshape(-1, 1)
    x_kernel = partial(rbf_kernel, gamma=cfg['hyperparams']['x_kernel_gamma'])
    y_kernel = partial(rbf_kernel, gamma=cfg['hyperparams']['y_kernel_gamma'])
    print("Computing TSNE")

    X_reduced = TSNE(
        n_components=cfg['embedding']['n_components'],
        random_state=1990).fit_transform(X)

    print('Computing MCMD')
    mcmd_vals = compute_mcmd(
                grid=X_reduced,
                x=X_reduced,
                y=Y,
                x_prime=np.tile(X_reduced, (cfg['data']['n_draws'], 1)),
                y_prime=y_prime,
                x_kernel=x_kernel,
                y_kernel=y_kernel,
            )
    mean_mcmd = np.mean(mcmd_vals)
    total_mcmd = np.sum(mcmd_vals)
    print("Average MCMD: {:.4f}".format(mean_mcmd))
    print("Total MCMD: {:.4f}".format(total_mcmd))

    if cfg['plot']['gen_fig']:

        if cfg['embedding']['n_components'] != 2:
            raise ValueError("Can only plot 2D embeddings.")

        get_scatter_plot_by_cls(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            c=Y.flatten(),
            fpath="probcal/figures/artifacts/mnist_tsne.png",
            title="MNIST TSNE"
        )

        plot_hex_bin_mcmd(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            c=mcmd_vals,
            title="MCMD on MNIST: {} DNN ({:.3f})".format(cfg['model_type'], total_mcmd),
            grid_size=cfg['plot']['grid_size'],
            fpath=f"probcal/figures/artifacts/mnist_{cfg['model_type']}_mcmd.png",
            crange=(cfg['plot']['color_min'], cfg['plot']['color_max'])
        )

if __name__ == "__main__":
    cfg = yaml.safe_load(open("probcal/experiments/mnist_calib_cfg.yaml", "r"))
    main(cfg)