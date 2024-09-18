import torch

def polynomial_kernel_torch(x: torch.Tensor, x_prime: torch.Tensor, gamma: float | None = None, coef0: float = 1.,
                            degree: int = 3):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    K = torch.matmul(x, x_prime.T)
    K *= gamma
    K += coef0
    K **= degree
    return K


def rbf_kernel(x: torch.Tensor, x_prime: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """Pytorch implementation of sklearn's RBF kernel.

    Args:
        x (torch.Tensor): A (n,) or (n,d) feature tensor.
        x_prime (torch.Tensor): A (m,) or (m,d) feature tensor.
        gamma (float | None, optional): Gamma parameter for the kernel. If None, defaults to 1.0 / d.

    Raises:
        ValueError: If x and x_prime do not have the same number of dimensions.
        ValueError: If x and x_prime do not have the same feature dimension.

    Returns:
        torch.Tensor: (n,m) Gram tensor.
    """
    if x.ndim != x_prime.ndim:
        raise ValueError("x and x_prime must have same number of dimensions.")

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x_prime.ndim == 1:
        x_prime = x_prime.reshape(-1, 1)

    if x.shape[-1] != x_prime.shape[-1]:
        raise ValueError("x and x_prime must have same feature dimension.")

    gamma = gamma or 1.0 / x.shape[-1]
    K = torch.exp(-gamma * torch.cdist(x, x_prime, p=2) ** 2)
    return K