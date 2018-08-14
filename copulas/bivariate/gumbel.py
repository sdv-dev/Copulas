import sys

import numpy as np
from scipy.optimize import fminbound

from copulas.bivariate.base import Bivariate, CopulaTypes


class Gumbel(Bivariate):
    """Class for clayton copula model."""

    copula_type = CopulaTypes.GUMBEL

    def get_generator(self):
        """Return the generator function."""
        def generator(theta, t):
            return np.power(-np.log(t), theta)

        return generator

    def probability_density(self, U, V):
        """Compute density function for given copula family."""
        if self.theta < 1:
            raise ValueError("Theta cannot be less than 1 for Gumbel")

        elif self.theta == 1:
            return np.multiply(U, V)

        else:
            a = np.power(np.multiply(U, V), -1)
            tmp = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
            b = np.power(tmp, -2 + 2.0 / self.theta)
            c = np.power(np.multiply(np.log(U), np.log(V)), self.theta - 1)
            d = 1 + (self.theta - 1) * np.power(tmp, -1.0 / self.theta)
            return self.copula_cumulative_density(U, V) * a * b * c * d

    def copula_cumulative_density(self, U, V):
        """Computes the cumulative distribution function for the copula, :math:`C(u, v)`

        Args:
            U: `np.ndarray`
            V: `np.ndarray`

        Returns:
            np.array: cumulative probability
        """
        if self.theta < 1:
            raise ValueError("Theta cannot be less than 1 for Gumbel")

        elif self.theta == 1:
            return np.multiply(U, V)

        else:
            h = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
            h = -np.power(h, 1.0 / self.theta)
            cdfs = np.exp(h)
            return cdfs

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative density :math:`C(u|v)^-1`

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """

        if self.theta == 1:
            return y

        else:
            u = fminbound(
                self.partial_derivative_cumulative_density,
                sys.float_info.epsilon,
                1.0,
                args=(V, y)
            )
            return u

    def partial_derivative_cumulative_density(self, U, V, y=0):
        """Compute partial derivative :math:`C(u|v)` of cumulative density.

        Args:
            U: `np.ndarray`
            V: `np.ndarray`
            y: `float`

        Returns:

        """
        if self.theta == 1:
            return V

        else:
            t1 = np.power(-np.log(U), self.theta)
            t2 = np.power(-np.log(V), self.theta)
            p1 = np.exp(-np.power((t1 + t2), 1.0 / self.theta))
            p2 = np.power(t1 + t2, -1 + 1.0 / self.theta)
            p3 = np.power(-np.log(U), self.theta - 1)
            return np.divide(np.multiply(np.multiply(p1, p2), p3), U) - y

    def get_theta(self):
        """Compute theta parameter using Kendall's tau.

        On Gumbel copula :math:\\tau is defined as :math:`τ = \\frac{θ−1}{θ}`
        that we solve as :math:`θ = \\frac{1}{1-τ}`
        """
        if self.tau == 1:
            theta = 1000
        else:
            theta = 1 / (1 - self.tau)

        return theta

    def _sample(self, v, c):
        u = np.empty([1, 0])

        for v_, c_ in zip(v, c):
            u = np.append(u, self.percent_point(v_, c_))

        return u
