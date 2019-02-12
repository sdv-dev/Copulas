
import numpy as np
from scipy.optimize import fminbound

from copulas import EPSILON
from copulas.bivariate.base import Bivariate, CopulaTypes


class Gumbel(Bivariate):
    """Class for clayton copula model."""

    copula_type = CopulaTypes.GUMBEL
    theta_interval = [1, float('inf')]
    invalid_thetas = []

    def generator(self, t):
        """Return the generator function."""
        return np.power(-np.log(t), self.theta)

    def probability_density(self, X):
        """Compute density function for given copula family."""
        self.check_fit()

        U, V = self.split_matrix(X)

        if self.theta == 1:
            return np.multiply(U, V)

        else:
            a = np.power(np.multiply(U, V), -1)
            tmp = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
            b = np.power(tmp, -2 + 2.0 / self.theta)
            c = np.power(np.multiply(np.log(U), np.log(V)), self.theta - 1)
            d = 1 + (self.theta - 1) * np.power(tmp, -1.0 / self.theta)
            return self.cumulative_distribution(X) * a * b * c * d

    def cumulative_distribution(self, X):
        """Computes the cumulative distribution function for the copula, :math:`C(u, v)`

        Args:
            X: `np.ndarray`

        Returns:
            np.array: cumulative probability
        """
        self.check_fit()

        U, V = self.split_matrix(X)

        if self.theta == 1:
            return np.multiply(U, V)

        else:
            h = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
            h = -np.power(h, 1.0 / self.theta)
            cdfs = np.exp(h)
            return cdfs

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^-1`

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """

        self.check_fit()

        if self.theta == 1:
            return y

        else:
            result = []
            for _y, _V in zip(y, V):
                result.append(fminbound(
                    self.partial_derivative_scalar, EPSILON, 1.0, args=(_y, _V)
                ))

            return np.array(result)

    def partial_derivative(self, X, y=0):
        """Compute partial derivative :math:`C(u|v)` of cumulative density.

        Args:
            X: `np.ndarray`
            y: `float`

        Returns:

        """
        self.check_fit()

        U, V = self.split_matrix(X)

        if self.theta == 1:
            return V

        else:
            t1 = np.power(-np.log(U), self.theta)
            t2 = np.power(-np.log(V), self.theta)
            p1 = self.cumulative_distribution(X)
            p2 = np.power(t1 + t2, -1 + 1.0 / self.theta)
            p3 = np.power(-np.log(V), self.theta - 1)
            return np.divide(np.multiply(np.multiply(p1, p2), p3), V) - y

    def compute_theta(self):
        """Compute theta parameter using Kendall's tau.

        On Gumbel copula :math:`\\tau is defined as :math:`τ = \\frac{θ−1}{θ}`
        that we solve as :math:`θ = \\frac{1}{1-τ}`
        """
        if self.tau == 1:
            theta = 10000
        else:
            theta = 1 / (1 - self.tau)

        return theta
