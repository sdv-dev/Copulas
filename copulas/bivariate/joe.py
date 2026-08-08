import numpy as np
from copulas.bivariate.base import Bivariate, CopulaTypes
from copulas.bivariate.utils import split_matrix

class Joe(Bivariate):
    """Class for Joe copula model."""

    copula_type = CopulaTypes.JOE
    theta_interval = [0, float('inf')]
    invalid_thetas = [0]

    def __init__(self, theta=1):
        """
        Initialize a Joe copula.

        Parameters:
        - theta (float): The theta parameter of the Joe copula. Default is 1.
        """
        self.theta = theta

    def generator(self, t):
        """
        Compute the generator function of the Joe copula.

        The generator function of the Joe copula is defined as:

        .. math:: G(t) = -((1 - t^(-theta))^(-1/theta))

        Parameters:
        - t (float or np.array): The value(s) at which to evaluate the generator.

        Returns:
        - float or np.array: The value(s) of the generator function at t.
        """
        return -((1 - t ** (-self.theta)) ** (1 / self.theta))

    def pdf(self, X):
        """
        Compute the probability density function for the Joe copula.

        The probability density function (PDF) for the Joe copula is given by:

        .. math:: c(u, v) = (theta - 1) * (u^(-theta) + v^(-theta) - 1)^(theta - 2) * u^(-theta - 1) * v^(-theta - 1)

        Parameters:
        - X (np.array): The input array of shape (n, 2) containing pairs of values (u, v).

        Returns:
        - np.array: The probability density values for each pair in X.
        """
        U, V = split_matrix(X)
        return (self.theta - 1) * (U ** (-self.theta) + V ** (-self.theta) - 1) ** (self.theta - 2) * U ** (-self.theta - 1) * V ** (-self.theta - 1)

    def cdf(self, X):
        """
        Compute the cumulative distribution function for the Joe copula.

        The cumulative distribution function (CDF) for the Joe copula is given by:

        .. math:: C(u, v) = (u^(-theta) + v^(-theta) - 1)^theta

        Parameters:
        - X (np.array): The input array of shape (n, 2) containing pairs of values (u, v).

        Returns:
        - np.array: The cumulative distribution values for each pair in X.
        """
        U, V = split_matrix(X)
        return (U ** (-self.theta) + V ** (-self.theta) - 1) ** self.theta

    def percent_point(self, y, V):
        """
        Compute the inverse of conditional cumulative distribution :math:`C(u|v)^{-1}`.

        The inverse of conditional cumulative distribution :math:`C(u|v)^{-1}` for the Joe copula
        is given by:

        .. math:: (y^(1/theta) + v^(-theta) - 1)^(-1/theta)

        Parameters:
        - y (float or np.array): The value(s) of :math:`C(u|v)`.
        - V (float or np.array): The given value(s) of v.

        Returns:
        - float or np.array: The inverse of conditional cumulative distribution values.
        """
        return (y ** (1 / self.theta) + V ** (-self.theta) - 1) ** (-1 / self.theta)
