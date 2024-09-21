import torch

class MixUpTransform:
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.lmbda * x1 + (1 - self.lmbda) * x2
