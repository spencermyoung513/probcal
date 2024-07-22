import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from probcal.utils.experiment_utils import get_model, get_datamodule
from probcal.utils.configs import TrainingConfig



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
Y_hat = np.zeros(len(test_loader))




for i, (x, y) in enumerate(test_loader):
    X[i] = x.flatten()
    Y[i] = y
    with torch.no_grad():
        y_hat = model._predict_impl(x).squeeze().detach().numpy()


X_embedded = TSNE(n_components=1).fit_transform(X)
plt.scatter(X_embedded, Y)
plt.show()






