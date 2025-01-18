import torch

from probcal.utils.differentiable_samplers import get_differentiable_sample_from_gaussian
from probcal.utils.differentiable_samplers import get_differentiable_sample_from_nbinom
from probcal.utils.differentiable_samplers import get_differentiable_sample_from_poisson


def test_gaussian_sample_shape_matches_expectation():
    mu = torch.zeros((6, 1))
    stdev = torch.ones((6, 1))

    samples = get_differentiable_sample_from_gaussian(mu, stdev, num_samples=3)

    assert samples.shape == (3, 6, 1)


def test_gaussian_sample_statistics():
    mu = torch.randn((6, 1))
    stdev = torch.abs(torch.randn((6, 1))) + 1e-3

    samples = get_differentiable_sample_from_gaussian(mu, stdev, num_samples=10000)

    empirical_mean = samples.mean(dim=0)
    empirical_std = samples.std(dim=0)

    assert torch.allclose(empirical_mean, mu, atol=1e-1)
    assert torch.allclose(empirical_std, stdev, atol=1e-1)


def test_poisson_sample_shape_matches_expectation():
    lmbda = torch.randint(low=1, high=8, size=(6, 1))
    samples = get_differentiable_sample_from_poisson(lmbda, num_samples=3)
    assert samples.shape == (3, 6, 1)


def test_poisson_sample_statistics():
    lmbda = torch.randint(low=1, high=8, size=(6, 1))
    samples = get_differentiable_sample_from_poisson(lmbda, num_samples=10000, temperature=0.001)

    empirical_mean = samples.mean(dim=0)
    empirical_var = samples.var(dim=0)
    assert torch.allclose(empirical_mean, lmbda.float(), atol=0.5, rtol=0.1)
    assert torch.allclose(empirical_var, lmbda.float(), atol=0.5, rtol=0.1)


def test_nbinom_sample_shape_matches_expectation():
    mu = torch.rand(size=(4, 1)) * 6
    alpha = torch.rand(size=(4, 1))
    samples = get_differentiable_sample_from_nbinom(mu, alpha, num_samples=3)
    assert samples.shape == (3, 4, 1)


def test_nbinom_sample_statistics():
    mu = torch.rand(size=(4, 1)) * 6
    alpha = torch.rand(size=(4, 1))
    var = mu + (mu**2) * alpha
    samples = get_differentiable_sample_from_nbinom(
        mu, alpha, num_samples=10000, temperature=0.001
    )

    empirical_mean = samples.mean(dim=0)
    empirical_var = samples.var(dim=0)
    assert torch.allclose(empirical_mean, mu, rtol=0.1, atol=0.5)
    assert torch.allclose(empirical_var, var, rtol=0.1, atol=0.5)
