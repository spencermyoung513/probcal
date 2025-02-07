import torch

from probcal.evaluation.metrics import compute_lce


def _kronecker_kernel(x: torch.Tensor, x_prime: torch.Tensor):
    x = x.unsqueeze(1)
    x_prime = x_prime.unsqueeze(0)
    return (x == x_prime).float()


def test_compute_lce_works_as_expected():
    x = torch.tensor([1, 2, 3])
    X = torch.tensor([1, 2, 3])
    Y = torch.tensor([0, 1, 2])
    i_cdf = torch.distributions.Normal(loc=torch.tensor([0.0, 1.0, 2.0]), scale=torch.ones(3)).icdf
    kernel = _kronecker_kernel
    num_bins = 2
    lce = compute_lce(x=x, X=X, Y=Y, i_cdf=i_cdf, kernel=kernel, num_bins=num_bins)

    expected_lce = torch.tensor([0.125, 0.125, 0.125])
    assert torch.allclose(lce, expected_lce)
