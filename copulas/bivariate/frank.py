
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fminbound, fsolve

from copulas import EPSILON
from copulas.bivariate.base import Bivariate, CopulaTypes


class Frank(Bivariate):
    """Class for Frank copula model."""

    copula_type = CopulaTypes.FRANK
    theta_interval = [-float('inf'), float('inf')]
    invalid_thetas = [0]

    def generator(self, t):
        """Return the generator function."""
        a = (np.exp(-self.theta * t) - 1) / (np.exp(-self.theta) - 1)
        return -np.log(a)

    def _g(self, z):
        """Helper function to solve Frank copula.

        This functions encapsulates :math:`g_z = e^{-\\theta z} - 1` used on Frank copulas.

        Argument:
            z: np.ndarray

        Returns:
            np.ndarray
        """
        return np.exp(np.multiply(-self.theta, z)) - 1

    def probability_density(self, X):
        """Compute density function for given copula family.

        Args:
            X: `np.ndarray`

        Returns:
            np.array: probability density
        """
        self.check_fit()

        U, V = self.split_matrix(X)

        if self.theta == 0:
            return np.multiply(U, V)

        else:
            num = np.multiply(np.multiply(-self.theta, self._g(1)), 1 + self._g(np.add(U, V)))
            aux = np.multiply(self._g(U), self._g(V)) + self._g(1)
            den = np.power(aux, 2)
            return num / den

    def cumulative_distribution(self, X):
        """Computes the cumulative distribution function for the copula, :math:`C(u, v)`

        Args:
            X: `np.ndarray`

        Returns:
            np.array: cumulative distribution
        """
        self.check_fit()

        U, V = self.split_matrix(X)

        num = np.multiply(
            np.exp(np.multiply(-self.theta, U)) - 1,
            np.exp(np.multiply(-self.theta, V)) - 1
        )
        den = np.exp(-self.theta) - 1

        return -1.0 / self.theta * np.log(1 + num / den)

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^-1`

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """
        self.check_fit()

        if self.theta < 0:
            return V

        else:
            result = []
            for _y, _V in zip(y, V):
                result.append(fminbound(
                    self.partial_derivative_scalar, EPSILON, 1.0, args=(_y, _V)
                ))

            return np.array(result)

    def partial_derivative(self, X, y=0):
        """Compute partial derivative :math:`C(u|v)` of cumulative distribution.

        Args:
            X: `np.ndarray`
            y: `float`

        Returns:
            np.ndarray
        """
        self.check_fit()

        U, V = self.split_matrix(X)

        if self.theta == 0:
            return V

        else:
            num = np.multiply(self._g(U), self._g(V)) + self._g(U)
            den = np.multiply(self._g(U), self._g(V)) + self._g(1)
            return (num / den) - y

    def compute_theta(self):
        """Compute theta parameter using Kendall's tau.

        On Frank copula, this is
        :math:`τ = 1 − \\frac{4}{θ} + \\frac{4}{θ^2}\\int_0^θ \\!
        \\frac{t}{e^t -1} \\, \\mathrm{d}t`.

        """
        return fsolve(self._frank_help, 1, args=(self.tau))[0]

    @staticmethod
    def _frank_help(alpha, tau):
        """Compute first order debye function to estimate theta."""

        def debye(t):
            return t / (np.exp(t) - 1)

        debye_value = integrate.quad(debye, EPSILON, alpha)[0] / alpha
        return 4 * (debye_value - 1) / alpha + 1 - tau
