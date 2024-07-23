import torch
import numpy as np
from functools import partial

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel

from probcal.utils.experiment_utils import get_model, get_datamodule
from probcal.utils.configs import TrainingConfig

from probcal.samplers import GaussianSampler
from probcal.evaluation.metrics import compute_mcmd


from probcal.enums import DatasetType, ImageDatasetName

datamodule = get_datamodule(
    DatasetType.IMAGE,
    ImageDatasetName.MNIST,
    1
)

datamodule.setup("predict")
test_loader = datamodule.test_dataloader()

cfg = TrainingConfig.from_yaml("../../configs/mnist_gaussian_cfg.yaml")
model = get_model(cfg)

state_dict = torch.load(
    "/Users/porterjenkins/code/probcal/weights/mnist_gaussian/best_mae_gaussian.pt",
    map_location='cpu'
)
model.load_state_dict(state_dict)



X = np.zeros((len(test_loader), 28*28))
Y = np.zeros(len(test_loader))
Y_hat = torch.zeros((len(test_loader), 2))


for i, (x, y) in enumerate(test_loader):
    X[i] = x.flatten()
    Y[i] = y
    with torch.no_grad():
        y_hat = model._predict_impl(x).squeeze()
        Y_hat[i] = y_hat

sampler = GaussianSampler(yhat=Y_hat)

m = 10
y_prime = sampler.sample(m=m)
x_kernel = partial(rbf_kernel, gamma=0.5)
y_kernel = partial(rbf_kernel, gamma=0.5)
print("Computing TSNE")
X_reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
print('Computing MCMD')

mcmd_vals = compute_mcmd(
            grid=X_reduced,
            x=X_reduced,
            y=Y,
            x_prime=np.tile(X_reduced, m),
            y_prime=y_prime,
            x_kernel=x_kernel,
            y_kernel=y_kernel,
        )

print(mcmd_vals)
print(np.mean(model))
