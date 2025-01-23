from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rotate


if __name__ == "__main__":
    root_dir = Path("data/rotated-mnist")
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    test_dir = root_dir / "test"

    angle_generator = np.random.default_rng(seed=1998)

    for dir in train_dir, val_dir, test_dir:
        dir.mkdir(parents=True, exist_ok=True)

    mnist_train = MNIST(root="data", train=True, download=True, transform=ToTensor())
    mnist_test = MNIST(root="data", train=False, download=True, transform=ToTensor())

    for i, (img, _) in enumerate(mnist_train):
        angle = angle_generator.uniform(-np.pi, np.pi)
        rotated_img = rotate(img, angle * (180 / np.pi))
        torch.save((rotated_img, angle), train_dir / f"{i}.pt")

    shuffled_test_indices = np.random.default_rng(seed=1998).permutation(10000)

    for i, idx in enumerate(shuffled_test_indices[:5000]):
        img = mnist_test[idx][0]
        angle = angle_generator.uniform(-np.pi, np.pi)
        rotated_img = rotate(img, angle * (180 / np.pi))
        torch.save((rotated_img, angle), val_dir / f"{i}.pt")

    for i, idx in enumerate(shuffled_test_indices[5000:]):
        img = mnist_test[idx][0]
        angle = angle_generator.uniform(-np.pi, np.pi)
        rotated_img = rotate(img, angle * (180 / np.pi))
        torch.save((rotated_img, angle), test_dir / f"{i}.pt")
