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

    def get_pdf(self):
        """Compute density function for given copula family."""
        def pdf(U, V):
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
                return self.get_cdf()(U, V) * a * b * c * d

        return pdf

    def get_cdf(self):
        """Compute cdf function for given copula family."""
        def cdf(U, V):
            if self.theta < 1:
                raise ValueError("Theta cannot be less than 1 for Gumbel")

            elif self.theta == 1:
                return np.multiply(U, V)

            else:
                h = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
                h = -np.power(h, 1.0 / self.theta)
                cdfs = np.exp(h)
                return cdfs

        return cdf

    def get_ppf(self):
        """Compute the inverse of conditional CDF C(u|v)^-1.

        Args:
            v : given value of v
            y: value of C(u|v)
        """
        def ppf(v, y, theta):
            if theta == 1:
                return y

            else:
                dev = self.get_h_function()
                u = fminbound(dev, sys.float_info.epsilon, 1.0, args=(v, y, theta))
                return u

        return ppf

    def get_h_function(self):
        """Compute partial derivative C(u|v) of each copula cdf function."""
        def du(u, v, theta, y=0):
            if theta == 1:
                return v

            else:
                t1 = np.power(-np.log(u), theta)
                t2 = np.power(-np.log(v), theta)
                p1 = np.exp(-np.power((t1 + t2), 1.0 / theta))
                p2 = np.power(t1 + t2, -1 + 1.0 / theta)
                p3 = np.power(-np.log(u), theta - 1)
                result = np.divide(np.multiply(np.multiply(p1, p2), p3), u)
                result = result - y
                return result

        return du

    def get_theta(self):
        """Compute theta parameter using Kendall's tau.

        For Gumbel copula we have :math:`τ = \\frac{θ−1}{θ}` that we solve as :math:`θ = \\frac{1}{1-τ}`
        """
        if self.tau == 1:
            theta = 1000
        else:
            theta = 1 / (1 - self.tau)

        return theta

    def _sample(self, v, c):
        u = np.empty([1, 0])
        ppf = self.get_ppf()

        for v_, c_ in zip(v, c):
            u = np.append(u, ppf(v_, c_, self.theta))

        return u
