"""A compilation of pytorch-native kernels."""
import torch


def l2_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise L2 distance matrix between rows of input tensors x and y.

    Args:
        x: Tensor of shape (n, d) where n is the number of points in the first tensor and d is the dimensionality.
        y: Tensor of shape (m, d) where m is the number of points in the second tensor and d is the dimensionality.

    Returns:
        Tensor of shape (n, m) where each element (i, j) is the L2 distance between x[i] and y[j].
    """
    # Compute the squared norm of each row vector in x and y
    squared_norms_x = torch.sum(x**2, dim=1, keepdim=True)  # Shape (n, 1)
    squared_norms_y = torch.sum(y**2, dim=1, keepdim=True)  # Shape (m, 1)

    # Compute the pairwise squared distance matrix
    pairwise_squared_distances = squared_norms_x + squared_norms_y.t() - 2 * torch.mm(x, y.t())

    # Add small epsilon for numerical stability and take square root to get L2 distance
    pairwise_distances = torch.sqrt(torch.clamp(pairwise_squared_distances, min=1e-12))

    return pairwise_distances


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
    K = torch.exp(-gamma * l2_norm(x, x_prime) ** 2)
    return K


def laplacian_kernel(
    x: torch.Tensor, x_prime: torch.Tensor, gamma: float | None = None
) -> torch.Tensor:
    """Pytorch implementation of sklearn's laplacian kernel.

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
    K = torch.exp(-gamma * torch.cdist(x, x_prime, p=1) ** 2)
    return K


def polynomial_kernel(
    x: torch.Tensor,
    x_prime: torch.Tensor,
    gamma: float | None = None,
    coef0: float = 1.0,
    degree: int = 3,
) -> torch.Tensor:
    """Pytorch implementation of sklearn's polynomial kernel.

    Args:
        x (torch.Tensor): A (n,) or (n,d) feature tensor.
        x_prime (torch.Tensor): A (m,) or (m,d) feature tensor.
        gamma (float | None, optional): Gamma parameter for the kernel. If None, defaults to 1.0 / d.
        coef0 (float, optional): Coef0 parameter for the kernel (added to the inner product before applying the exponent). Defaults to 1.
        degree (int, optional): Degree of the polynomial kernel. Defaults to 3.

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

    if gamma is None:
        gamma = 1.0 / x.shape[1]
    K = torch.matmul(x, x_prime.T)
    K *= gamma
    K += coef0
    K **= degree
    return K
