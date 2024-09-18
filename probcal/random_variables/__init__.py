from scipy.stats import norm
from .double_poisson import DoublePoisson
from ..enums import HeadType

RVS = {
    HeadType.DOUBLE_POISSON.value: DoublePoisson,
    HeadType.GAUSSIAN.value: norm,
}