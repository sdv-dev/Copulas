import sys

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fminbound, fsolve

from copulas.bivariate import Bivariate


class Frank(Bivariate):
    """ Class for clayton copula model """

    def __init__(self):
        super(Frank, self).__init__()

    def get_generator(self):
        """Return the generator function.
        """
        def generator(theta, t):
            a = (np.exp(-theta * t) - 1) / (np.exp(-theta) - 1)
            return -np.log(a)
        return generator

    def get_pdf(self):
        """compute density function for given copula family
        """
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
        """Compute cdf function for given copula family
        """
        def cdf(U, V):
            if self.theta == 0:
                return np.multiply(U, V)
            else:
                num = np.multiply(
                    np.exp(np.multiply(-self.theta, U)) - 1,
                    np.exp(np.multiply(-self.theta, V)) - 1)
                den = np.exp(-self.theta) - 1
                return -1.0 / self.theta * np.log(1 + num / den)
        return cdf

    def get_ppf(self):
        """compute the inverse of conditional CDF C(u|v)^-1
        Args:
            y: value of C(u|v)
            v : given value of v
        """
        def ppf(y, v, theta):
            if theta < 0:
                return v
            else:
                dev = self.get_h_function()
                u = fminbound(dev, sys.float_info.epsilon, 1.0, args=(v, theta, y))
                return u
        return ppf

    def get_h_function(self):
        """Compute partial derivative C(u|v) of each copula cdf function
        :param theta: single parameter of the Archimedean copula
        :param cname: name of the copula function
        """
        def du(u, v, theta, y=0):
            if theta == 0:
                return v
            else:

                def g(theta, z):
                    return -1 + np.exp(-np.dot(theta, z))

                num = np.multiply(g(u, theta), g(v, theta)) + g(v, theta)
                den = np.multiply(g(u, theta), g(v, theta)) + g(1, theta)
                result = num / den
                result = result - y
                return result
        return du

    def tau_to_theta(self):
        theta = fsolve(Frank._frank_help, 1, args=(self.tau))[0]
        return theta

    @staticmethod
    def _frank_help(alpha, tau):
        """compute first order debye function to estimate theta
        """

        def debye(t):
            return t / (np.exp(t) - 1)

        debye_value = integrate.quad(debye, sys.float_info.epsilon, alpha)[0] / alpha
        return 4 * (debye_value - 1) / alpha + 1 - tau
