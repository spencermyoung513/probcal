from typing import Union

from abc import ABC, abstractmethod

import numpy as np
import scipy
import torch


class BaseSampler(ABC):

    def __init__(self, yhat: torch.Tensor):
        self.n = yhat.shape[0]
        self.dist = self.yhat_to_rvs(yhat)

    @abstractmethod
    def yhat_to_rvs(self, yhat: torch.Tensor) -> Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete]:
        pass

    @abstractmethod
    def get_nll(self, samples):
        pass

    def sample(self, m: int) -> np.ndarray:
        """
        Draw samples from the underlying scipy rvs object.
        Args:
            m: (int) number of samples to draw

        Returns: (np.ndarray) samples of shape (m, n) where n is the number of model predictions (y_hat) and m is the
        number of somples for each model output

        """
        draws = self.dist.rvs(size=(m, self.n))
        return draws