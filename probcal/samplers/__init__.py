from probcal.enums import HeadType

from .gaussian_sampler import GaussianSampler
from .poisson_sampler import PoissonSampler
from .nbinom_sampler import NegBinomSampler

SAMPLERS = {
    HeadType.GAUSSIAN.value: GaussianSampler,
    HeadType.POISSON.value: PoissonSampler,
    HeadType.NEGATIVE_BINOMIAL.value: NegBinomSampler
}