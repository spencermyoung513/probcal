from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor


class CIFAR100DataModule(L.LightningDataModule):

    IMG_SIZE = 32
    CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR100_STD = [0.2673, 0.2564, 0.2762]

    def __init__(
        self, root_dir: str | Path, batch_size: int, num_workers: int, persistent_workers: bool
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage: str):
        transform = Compose([ToTensor(), Normalize(self.CIFAR100_MEAN, self.CIFAR100_STD)])
        self.cifar100_test = CIFAR100(
            self.root_dir, train=False, download=True, transform=transform
        )
        self.cifar100_predict = CIFAR100(
            self.root_dir, train=False, download=True, transform=transform
        )
        cifar100_full = CIFAR100(self.root_dir, train=True, download=True, transform=transform)
        self.cifar100_train, self.cifar100_val = random_split(cifar100_full, [45000, 5000])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar100_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar100_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar100_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.cifar100_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )


# # test the datamodule out
# dm = CIFAR100DataModule(root_dir="data", batch_size=32, num_workers=4, persistent_workers=True)
# dm.setup("fit")

# # print a image out
# import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# import torchvision.transforms.functional as F

# # Load the meta file
# with open('data/cifar-100-python/meta', 'rb') as f:
#     meta = pickle.load(f, encoding='latin1')

# # Extract the fine label names
# fine_label_names = meta['fine_label_names']

# # get some random training images
# test_loader = dm.test_dataloader()
# for batch in test_loader:
#     images, labels = batch
#     #save the image as a png using matplotlib
#     print(images[0].shape)
#     print(np.min(images[0].numpy()))
#     print(np.max(images[0].numpy()))
#     plt.imshow(np.transpose(images[1].numpy(), (1, 2, 0)))
#     plt.savefig(f"test_image_{1}.png")

#     # Resize the image to 224x224
#     resized_image = F.resize(images[1], [224, 224])
#     plt.imshow(np.transpose(resized_image.numpy(), (1, 2, 0)))
#     plt.savefig(f"resized_test_image_{1}.png")

#     print(fine_label_names[labels[1].item()])
#     break
