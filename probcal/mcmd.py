from typing import Callable, Optional
import numpy as np



class MCMD(object):

    def __init__(
            self,
            num_x: int,
            num_x_prime: int,
            grid: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            x_prime: np.ndarray,
            y_prime: np.ndarray,
            x_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
            y_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
            lmbda: Optional[float] = 0.01
    ):
        self.n = num_x
        self.m = num_x_prime
        self.grid = grid
        self.x = x
        self.y = y
        self. x_prime = x_prime
        self.y_prime = y_prime
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel
        self.lmbda = lmbda


    def _get_kernel_x(self):
        K_X = self.x_kernel(self.x, self.x)
        K_X_prime = self.x_kernel(self.x_prime, self.x_prime)

        return K_X, K_X_prime

    def _get_kernel_y(self):
        K_Y = self.y_kernel(self.y, self.y)
        K_Y_prime = self.y_kernel(self.y_prime, self.y_prime)
        K_Y_Y_prime = self.y_kernel(self.y, self.y_prime)

        return K_Y, K_Y_prime, K_Y_Y_prime

    def _get_w_x(self, K_X, K_X_prime):
        W_X = np.linalg.inv(K_X + self.n * self.lmbda * np.eye(self.n))
        W_X_prime = np.linalg.inv(K_X_prime + self.m * self.lmbda * np.eye(self.m))
        return W_X, W_X_prime

    def _get_kernel_x_grid(self):
        k_X = self.x_kernel(self.x, self.grid)
        k_X_prime = self.x_kernel(self.x_prime, self.grid)

        return k_X, k_X_prime

    def _get_first_term(self, k_X, W_X, K_Y):
        return np.diag(k_X.T @ W_X @ K_Y @ W_X.T @ k_X)

    def _get_second_term(self, k_X, W_X, K_Y_Y_prime, W_X_prime, k_X_prime):
        return np.diag(2 * k_X.T @ W_X @ K_Y_Y_prime @ W_X_prime.T @ k_X_prime)

    def _get_third_term(self, k_X_prime, W_X_prime, K_Y_prime):
        return np.diag(k_X_prime.T @ W_X_prime @ K_Y_prime @ W_X_prime.T @ k_X_prime)


    def compute_mcmd(self):
        K_X, K_X_prime = self._get_kernel_x()
        W_X, W_X_prime = self._get_w_x(K_X, K_X_prime)
        K_Y, K_Y_prime, K_Y_Y_prime = self._get_kernel_y()
        k_X, k_X_prime = self._get_kernel_x_grid()

        first_term = self._get_first_term(k_X, W_X, K_Y)
        second_term = self._get_second_term(k_X, W_X, K_Y_Y_prime, W_X_prime, k_X_prime)
        third_term = self._get_third_term(k_X_prime, W_X_prime, K_Y_prime)

        mcmd = first_term - second_term + third_term
        return mcmd


