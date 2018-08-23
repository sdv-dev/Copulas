
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fminbound, fsolve

from copulas import EPSILON
from copulas.bivariate.base import Bivariate, CopulaTypes


class Frank(Bivariate):
    """Class for Frank copula model."""

    copula_type = CopulaTypes.FRANK

    def get_generator(self):
        """Return the generator function."""
        def generator(theta, t):
            a = (np.exp(-theta * t) - 1) / (np.exp(-theta) - 1)
            return -np.log(a)

        return generator

    def get_pdf(self):
        """Compute density function for given copula family."""
        def pdf(U, V):
            if self.theta == 0:
                return np.multiply(U, V)

            else:
                def g(z, theta):
                    return np.exp(np.multiply(-theta, z)) - 1
                num = np.multiply(np.multiply(-self.theta, g(1, self.theta)),
                                  1 + g(np.add(U, V), self.theta))
                den = np.power(np.multiply(g(U, self.theta), g(V, self.theta)) +
                               g(1, self.theta), 2)
                return num / den

        return pdf

    def get_cdf(self):
        """Compute cdf function for given copula family."""
        def cdf(U, V):
            if self.theta < 0:
                raise ValueError("Theta cannot be less than 0 for Frank")
            elif self.theta == 0:
                return np.multiply(U, V)

            else:
                num = np.multiply(
                    np.exp(np.multiply(-self.theta, U)) - 1,
                    np.exp(np.multiply(-self.theta, V)) - 1)
                den = np.exp(-self.theta) - 1
                return -1.0 / self.theta * np.log(1 + num / den)

        return cdf

    def get_ppf(self):
        """Compute the inverse of conditional CDF C(u|v)^-1.

        Args:
            y: value of C(u|v)
            v : given value of v
        """
        def ppf(y, v, theta):
            if theta < 0:
                return v
            else:
                dev = self.get_h_function()
                return fminbound(dev, EPSILON, 1.0, args=(v, theta, y))

        return ppf

    def get_h_function(self):
        """Compute partial derivative C(u|v) of cdf function."""
        def du(u, v, theta, y=0):
            if theta == 0:
                return v

            else:

                def g(z, theta):
                    return -1 + np.exp(-np.multiply(theta, z))

                num = np.multiply(g(u, theta), g(v, theta)) + g(u, theta)
                den = np.multiply(g(u, theta), g(v, theta)) + g(1, theta)
                return (num / den) - y

        return du

    def tau_to_theta(self):
        return fsolve(Frank._frank_help, 1, args=(self.tau))[0]

    @staticmethod
    def _frank_help(alpha, tau):
        """Compute first order debye function to estimate theta."""

        def debye(t):
            return t / (np.exp(t) - 1)

        debye_value = integrate.quad(debye, EPSILON, alpha)[0] / alpha
        return 4 * (debye_value - 1) / alpha + 1 - tau

    def _sample(self, v, c):
        u = np.empty([1, 0])
        ppf = self.get_ppf()

        for v_, c_ in zip(v, c):
            u = np.append(u, ppf(v_, c_, self.theta))

        return u
