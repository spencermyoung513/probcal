from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt

from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.models import GaussianNN


data = np.load("data/discrete-wave/discrete_sine_wave.npz")
X_val = torch.tensor(data["X_val"]).flatten().float()
y_val = torch.tensor(data["y_val"]).flatten().float()
X_test = torch.tensor(data["X_test"]).flatten().float()
y_test = torch.tensor(data["y_test"]).flatten().float()

grid = X_test
extended_grid = torch.cat(
    [torch.linspace(-torch.pi, 0, 50), X_test, torch.linspace(2 * torch.pi, 3 * torch.pi, 50)]
)

model = GaussianNN.load_from_checkpoint("weights/discrete-wave/gaussian.ckpt")
y_hat = model.predict(X_val.view(-1, 1))
l = 3

x_samples = torch.repeat_interleave(
    X_test,
    repeats=l,
    dim=0,
)
y_samples = model.sample(y_hat, num_samples=l).flatten()

gamma_vals = [0.1, 0.5, 1, 2]
fig, axs = plt.subplots(
    len(gamma_vals), 1, figsize=(8, 3 * len(gamma_vals)), sharex="col", sharey="col"
)

for ax, gamma in zip(axs.ravel(), gamma_vals):
    x_kernel = partial(rbf_kernel, gamma=gamma)
    y_kernel = partial(rbf_kernel, gamma=1 / (2 * y_val.float().var().item()))
    mcmd_vals = compute_mcmd_torch(
        grid=extended_grid,
        x=X_test,
        y=y_test,
        x_prime=x_samples,
        y_prime=y_samples,
        x_kernel=x_kernel,
        y_kernel=y_kernel,
        lmbda=0.1,
    )
    order = torch.argsort(extended_grid)
    left_mask = extended_grid <= 0
    mid_mask = (extended_grid >= 0) & (extended_grid <= 2 * torch.pi)
    right_mask = extended_grid >= 2 * torch.pi
    ax.plot(
        extended_grid[order][left_mask],
        mcmd_vals[order][left_mask],
        color="tab:blue",
        linestyle="dashed",
    )
    ax.plot(
        extended_grid[order][right_mask],
        mcmd_vals[order][right_mask],
        color="tab:blue",
        linestyle="dashed",
    )
    ax.plot(extended_grid[order][mid_mask], mcmd_vals[order][mid_mask], color="tab:blue")
    ax.set_xlabel("$X$")
    ax.set_ylabel("CCE")
    ax.set_title(f"$\gamma_x = {gamma:.1f}$")

fig.tight_layout()
fig.savefig("probcal/figures/artifacts/cce_at_boundaries.pdf", dpi=150)
