import torch
from scipy import stats


def get_differentiable_sample_from_gaussian(
    mu: torch.Tensor, stdev: torch.Tensor, num_samples: int = 1
) -> torch.Tensor:
    """Use the reparametrization trick to sample from the specified gaussian in a differentiable manner.

    Args:
        mu (torch.Tensor): Mean of the gaussian to sample from. Shape: (n, 1).
        stdev (torch.Tensor): Standard deviation of the gaussian to sample from. Shape: (n, 1).
        num_samples (int, optional): Number of samples to draw. Defaults to 1.

    Returns:
        torch.Tensor: A (num_samples, n, 1) tensor of samples.
    """
    dist = torch.distributions.Normal(loc=mu, scale=stdev)
    return dist.rsample((num_samples,))


def get_differentiable_sample_from_poisson(
    lmbda: torch.Tensor, num_samples: int = 1, temperature: float = 0.1
) -> torch.Tensor:
    """Use a gamma reparametrization trick to sample from the specified poisson in a differentiable manner.

    Args:
        lmbda (torch.Tensor): Lambda parameter of the poisson to sample from. Shape: (n, 1).
        num_samples (int, optional): Number of samples to draw. Defaults to 1.
        temperature (float, optional): Temperature used for the differentiable relaxation. At 0, a perfect Poisson sample is recovered. Defaults to 0.1.

    Returns:
        torch.Tensor: A (num_samples, n, 1) tensor of samples.
    """
    num_events_to_simulate = max(
        10, int(stats.poisson.ppf(0.9999, mu=lmbda.long().detach().numpy()).max(axis=0).item())
    )
    z = torch.distributions.Exponential(lmbda.float()).rsample(
        (num_events_to_simulate, num_samples)
    )
    t = torch.cumsum(z, dim=0)
    relaxed_indicator = torch.sigmoid((1.0 - t) / temperature)
    N = relaxed_indicator.sum(dim=0)
    return N


def get_differentiable_sample_from_nbinom(
    mu: torch.Tensor, alpha: torch.Tensor, num_samples: int = 1, temperature: float = 0.1
) -> torch.Tensor:
    """Use the Poisson-Gamma mixture interpretation + the reparametrization trick to sample from the specified nbinom in a differentiable manner.

    Args:
        mu (torch.Tensor): Mu parameter of the negative binomial distribution to sample from. Shape: (n, 1).
        alpha (torch.Tensor): Alpha parameter of the negative binomial distribution to sample from. Shape: (n, 1).
        num_samples (int, optional): Number of samples to draw. Defaults to 1.
        temperature (float, optional): Temperature used for the differentiable relaxation. At 0, a perfect sample from the Poisson-Gamma is recovered. Defaults to 0.1.

    Returns:
        torch.Tensor: A (num_samples, n, 1) tensor of samples.
    """
    eps = torch.tensor(1e-6, device=mu.device)
    var = mu + alpha * mu**2
    p = mu / torch.maximum(var, eps)
    n = mu**2 / torch.maximum(var - mu, eps)

    lambda_sample = torch.distributions.Gamma(concentration=n, rate=p / (1 - p)).rsample(
        (num_samples,)
    )
    nbinom_sample = get_differentiable_sample_from_poisson(
        lambda_sample.flatten(), num_samples=1, temperature=temperature
    ).reshape(num_samples, *mu.shape)
    return nbinom_sample
