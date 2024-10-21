from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt

from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.models import GaussianNN
from probcal.models.regression_nn import RegressionNN


def compute_mae_and_nll_with_thresholded_reject(
    model: RegressionNN,
    threshold: float,
    thresholded_values: torch.Tensor,
    y_test: torch.Tensor,
    y_hat_test: torch.Tensor,
):
    mask = thresholded_values <= threshold
    if mask.float().sum() == 0:
        return 0.0, 0.0, 1.0
    y_test_pred = model.point_prediction(y_hat_test[mask], training=False).flatten()
    mae = (y_test[mask].flatten().float() - y_test_pred).abs().mean().detach()
    log_probs = model.posterior_predictive(y_hat_test[mask], training=False).log_prob(
        y_test[mask].flatten()
    )
    nll = -log_probs.mean()
    proportion_rejected = (~mask).float().mean()
    return mae, nll, proportion_rejected


def main():
    model = GaussianNN.load_from_checkpoint("weights/dispersed_waves_gaussian.ckpt")

    raw_data = np.load("data/dispersed_waves/dispersed_waves.npz")
    X_val = torch.tensor(raw_data["X_val"]).float().view(-1, 1)
    y_val = torch.tensor(raw_data["y_val"]).view(-1, 1)
    X_test = torch.tensor(raw_data["X_test"]).float().view(-1, 1)
    y_test = torch.tensor(raw_data["y_test"]).float().view(-1, 1)

    y_hat_val = model.predict(X_val)
    y_hat_test = model.predict(X_test)

    cce_mae_vals = []
    cce_nll_vals = []
    cce_reject_proportion_vals = []

    test_cce_vals = compute_mcmd_torch(
        grid=X_test,
        x=X_val,
        y=y_val.float(),
        x_prime=X_val,
        y_prime=model.sample(y_hat_val, num_samples=1).flatten().float(),
        x_kernel=partial(rbf_kernel, gamma=1),
        y_kernel=partial(rbf_kernel, gamma=1 / (2 * y_val.float().var())),
        lmbda=0.1,
    )

    cce_thresholds = torch.linspace(0.02, 0.06, steps=100)
    for threshold in cce_thresholds:
        mae, nll, proportion_rejected = compute_mae_and_nll_with_thresholded_reject(
            model,
            threshold=threshold,
            y_test=y_test,
            y_hat_test=y_hat_test,
            thresholded_values=test_cce_vals,
        )
        cce_reject_proportion_vals.append(proportion_rejected)
        cce_mae_vals.append(mae)
        cce_nll_vals.append(nll)

    cce_mae_vals = torch.tensor(cce_mae_vals)
    cce_nll_vals = torch.tensor(cce_nll_vals)
    cce_reject_proportion_vals = torch.tensor(cce_reject_proportion_vals)

    random_reject_proportion_vals = torch.linspace(0, cce_reject_proportion_vals.max(), steps=100)

    random_reject_mae_vals = []
    random_reject_nll_vals = []
    for p in random_reject_proportion_vals:
        mask = torch.randperm(len(y_test))[: int((1 - p) * len(y_test))]
        if len(mask) == 0:
            mae = 0
            nll = 0
        else:
            y_test_pred = model.point_prediction(y_hat_test[mask], training=False).flatten()
            mae = (y_test[mask].flatten().float() - y_test_pred).abs().mean().detach()
            log_probs = model.posterior_predictive(y_hat_test[mask], training=False).log_prob(
                y_test[mask].flatten()
            )
            nll = -log_probs.mean()
        random_reject_mae_vals.append(mae)
        random_reject_nll_vals.append(nll)

    random_reject_mae_vals = torch.tensor(random_reject_mae_vals)
    random_reject_nll_vals = torch.tensor(random_reject_nll_vals)

    cce_x = cce_reject_proportion_vals
    cce_y = cce_mae_vals
    random_x = random_reject_proportion_vals
    random_y = random_reject_mae_vals

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax: plt.Axes
    ax.plot(cce_x[:-1], cce_y[:-1], label="CCE-based rejection")
    ax.plot(random_x[:-1], random_y[:-1], label="Random rejection")
    ax.set_xlabel("Proportion Held Out")
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig("probcal/figures/artifacts/mae_with_rejection.pdf", dpi=150)

    cce_x = cce_reject_proportion_vals
    cce_y = cce_nll_vals
    random_x = random_reject_proportion_vals
    random_y = random_reject_nll_vals

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax: plt.Axes
    ax.plot(cce_x[:-1], cce_y[:-1], label="CCE-based rejection")
    ax.plot(random_x[:-1], random_y[:-1], label="Random rejection")
    ax.set_xlabel("Proportion Held Out")
    ax.set_ylabel("NLL")
    ax.legend()
    fig.tight_layout()
    fig.savefig("probcal/figures/artifacts/nll_with_rejection.pdf", dpi=150)


if __name__ == "__main__":
    main()
