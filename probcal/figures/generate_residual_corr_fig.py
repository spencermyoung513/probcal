from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy import stats

from probcal.evaluation.kernels import rbf_kernel
from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.models import GaussianNN

data = np.load("data/discrete-wave/discrete_sine_wave.npz")
X_val = torch.tensor(data["X_val"]).flatten().float()
y_val = torch.tensor(data["y_val"]).flatten().float()
X_test = torch.tensor(data["X_test"]).flatten().float()
y_test = torch.tensor(data["y_test"]).flatten().float()

grid = X_test

model = GaussianNN.load_from_checkpoint("weights/discrete-wave/gaussian.ckpt")
with torch.inference_mode():
    y_hat = model.predict(X_test.view(-1, 1))
l = 5

x_samples = torch.repeat_interleave(
    X_test,
    repeats=l,
    dim=0,
)
y_samples = model.sample(y_hat, num_samples=l).flatten()

x_kernel = partial(rbf_kernel, gamma=0.5)
y_kernel = partial(rbf_kernel, gamma=1 / (2 * y_val.float().var().item()))
mcmd_vals = compute_mcmd_torch(
    grid=grid,
    x=X_test,
    y=y_test,
    x_prime=x_samples,
    y_prime=y_samples,
    x_kernel=x_kernel,
    y_kernel=y_kernel,
    lmbda=0.1,
)

residuals = (y_hat[:, 0] - y_test).abs()
df = pd.DataFrame({"cce": mcmd_vals, "residuals": residuals})
r, p = stats.pearsonr(mcmd_vals, residuals)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.regplot(df, x="cce", y="residuals", line_kws={"color": "red"}, ax=ax)
ax.text(mcmd_vals.min(), residuals.max(), f"r = {r:.2f}, p = {p:.3g}", fontsize=12)
ax.set_xlabel("CCE")
ax.set_ylabel("$|y - \hat{\mu}|$")
fig.tight_layout()
fig.savefig("probcal/figures/artifacts/residual_corr.pdf", dpi=150)
