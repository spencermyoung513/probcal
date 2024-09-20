import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import torch
import open_clip

from probcal.utils.experiment_utils import get_model, get_datamodule
from probcal.utils.configs import TestConfig
from probcal.enums import DatasetType, ImageDatasetName, HeadType

from probcal.evaluation.metrics import compute_mcmd_torch
from probcal.evaluation.kernels import polynomial_kernel, rbf_kernel
from probcal.samplers import SAMPLERS
from probcal.random_variables import RVS

NUM_IMG_PLOT = 4
EXP_NAME = "ood_gaussian_blur_coco_people_gaussian"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", EXP_NAME)

if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

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

n = 100
m = 5
X = torch.zeros((n, 512)) # image embeddings
Y_true = torch.zeros((n, 1)) # true labels
Y_hat = []
imgs_to_plot = []
imgs_to_plot_preds = []
imgs_to_plot_true = []

for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
    with torch.no_grad():
        img_features = embedder.encode_image(x, normalize=True)
        pred = model._predict_impl(x)

    X[i] = img_features
    Y_true[i] = y
    Y_hat.append(pred)

    if i < NUM_IMG_PLOT:
        img = datamodule.denormalize(x)
        img = img.squeeze(0).permute(1, 2, 0).detach()
        imgs_to_plot.append(img)
        imgs_to_plot_preds.append(pred)
        imgs_to_plot_true.append(y)

    if i == (n-1):
        break

# plot images
fig, axs = plt.subplots(4, 2, figsize=(10, 8), sharey="col")
imgs_to_plot_preds = torch.cat(imgs_to_plot_preds, dim=0)
imgs_to_plot_true = torch.cat(imgs_to_plot_true, dim=0)
for i in range(NUM_IMG_PLOT):
    axs[i, 0].imshow(imgs_to_plot[i])
    axs[i, 0].set_title("Input Image")
    axs[i, 0].axis("off")

    rv = RVS[model_cfg.head_type.value](*imgs_to_plot_preds[i])
    disc_support = torch.arange(0, imgs_to_plot_true.max() + 5)
    dist_func = rv.pdf(disc_support)
    axs[i, 1].plot(disc_support, dist_func)
    axs[i, 1].scatter(imgs_to_plot_true[i], 0, color="black", marker="*", s=50, zorder=100)


plt.savefig(os.path.join(LOG_DIR, "input_images.png"))

# compute MCMD
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
        x_kernel=polynomial_kernel,
        y_kernel=partial(rbf_kernel, gamma=1 / (2 * Y_true.float().var())),
        lmbda=0.1,
    )


print(mcmd_vals.mean())