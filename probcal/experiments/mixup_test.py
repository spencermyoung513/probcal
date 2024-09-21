import torch
import matplotlib.pyplot as plt
from probcal.enums import DatasetType, ImageDatasetName
from probcal.utils.experiment_utils import get_model, get_datamodule

L = 0.25
def mixup_data(x1, x2, lam=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''



    mixed_x = lam * x1 + (1 - lam) * x2
    return mixed_x


datamodule = get_datamodule(
            DatasetType.IMAGE,
            ImageDatasetName.COCO_PEOPLE,
            batch_size=1,
            num_workers=0
        )

datamodule.setup(stage="test")
test_loader = datamodule.test_dataloader()

x1, y1 = test_loader.dataset[100]
x2, y2 = test_loader.dataset[50]

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharey="col")

x1 = datamodule.denormalize(x1.permute(1, 2, 0))
x2 = datamodule.denormalize(x2.permute(1, 2, 0))
x3 = mixup_data(x1, x2, L)
x4 = mixup_data(x2, x1, L)

axs[0].imshow(x1)
axs[0].set_title(f"Image: {1}")
axs[0].axis("off")

axs[1].imshow(x2)
axs[1].set_title(f"Image: {2}")
axs[1].axis("off")

axs[2].imshow(x3)
axs[2].set_title(f"MixUp: {L}")
axs[2].axis("off")

axs[3].imshow(x4)
axs[3].set_title(f"MixUp: {1-L}")
axs[3].axis("off")


plt.show()