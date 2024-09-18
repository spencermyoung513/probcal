import matplotlib.pyplot as plt
from functools import partial
import torch
import open_clip

from probcal.utils.experiment_utils import get_model, get_datamodule
from probcal.utils.configs import TestConfig
from probcal.enums import DatasetType, ImageDatasetName, HeadType

from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.kernels import polynomial_kernel_torch, rbf_kernel
from probcal.samplers import SAMPLERS

# build dataset and data loader
datamodule = get_datamodule(
        DatasetType.IMAGE,
        ImageDatasetName.COCO_PEOPLE,
        1,
        num_workers=0
    )
datamodule.setup(stage="test")
test_loader = datamodule.test_dataloader()

# instantiate model
model_cfg = TestConfig.from_yaml(f"configs/test/coco_gaussian_cfg.yaml")
model = get_model(model_cfg)
weights_fpath = "weights/coco_people_gaussian/version_0/state_dict_best_mae.ckpt"
state_dict = torch.load(weights_fpath, map_location='cpu')
model.load_state_dict(state_dict)

# get embeder
embedder, _, transform = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device="cpu",
)
embedder.eval()

n = 10
m = 5
X = torch.zeros((n, 512)) # image embeddings
Y_true = torch.zeros((n, 1)) # true labels
Y_hat = []

for i, (x, y) in enumerate(test_loader):
    print(x.shape, y.shape)
    with torch.no_grad():
        img_features = embedder.encode_image(x, normalize=True)
        pred = model._predict_impl(x)

    X[i] = img_features
    Y_true[i] = y
    Y_hat.append(pred)

    #img = datamodule.denormalize(x)
    #img = img.squeeze(0).permute(1, 2, 0).detach()
    #plt.imshow(img)
    #plt.show()
    if i == (n-1):
        break

Y_hat = torch.cat(Y_hat, dim=0)

sampler = SAMPLERS[model_cfg.head_type.value](Y_hat)
y_prime = sampler.sample(m=m).reshape(-1, 1)

with torch.inference_mode():
    x_prime = X.repeat_interleave(m, dim=0)
    print(x_prime.shape, y_prime.shape)

    mcmd_vals = compute_mcmd_torch(
        grid=X,
        x=X,
        y=Y_true.float(),
        x_prime=x_prime,
        y_prime=y_prime.float(),
        x_kernel=polynomial_kernel_torch,
        y_kernel=partial(rbf_kernel, gamma=1 / (2 * Y_true.float().var())),
        lmbda=0.1,
    )


print(mcmd_vals.mean())