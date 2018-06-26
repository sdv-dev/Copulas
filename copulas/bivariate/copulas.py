import logging

import numpy as np
import scipy.stats
from scipy.optimize import brentq, fmin

LOGGER = logging.getLogger(__name__)


class CopulaException(Exception):
    pass


class Copula(object):
    def __init__(self, U, V, theta=None, cname=None, dev=False):
        """Instantiates an instance of the copula object from a pandas dataframe

        :param data: the data matrix X
        :param utype: the distribution for the univariate, can be 'kde','norm'
        :param cname: the choice of copulas, can be 'clayton','gumbel','frank','gaussian'

        """
        self.U = U
        self.V = V
        self.theta = theta
        self.cname = cname
        if cname:
            if not theta:
                self.tau = scipy.stats.kendalltau(self.U, self.V)[0]
                self._get_parameter()
            self.pdf = self._get_pdf()
            self.cdf = self._get_cdf()
            self.ppf = self._get_ppf()
        if dev:
            self.derivative = self._get_du()

    def _get_parameter(self):
        """ estimate the parameter (theta) of copula given tau
        """

        if self.cname == 'clayton':
            if self.tau == 1:
                self.theta = 10000
            else:
                self.theta = 2 * self.tau / (1 - self.tau)

        elif self.cname == 'frank':
            self.theta = -fmin(self._frank_help, -5, disp=False)[0]

        elif self.cname == 'gumbel':
            if self.tau == 1:
                self.theta = 10000
            else:
                self.theta = 1 / (1 - self.tau)

    def _get_pdf(self):
        """compute density function for given copula family
        """
        if self.cname == 'clayton':
            def pdf(U, V, theta):
                if theta < 0:
                    raise ValueError("Theta cannot be than 0 for clayton")
                elif theta == 0:
                    return np.multiply(U, V)
                else:
                    a = (theta + 1) * np.power(np.multiply(U, V), -(theta + 1))
                    b = np.power(U, -theta) + np.power(V, -theta) - 1
                    c = -(2 * theta + 1) / theta
                    density = a * np.power(b, c)
                    return density
            return pdf

        elif self.cname == 'frank':
            def pdf(U, V, theta):
                if theta < 0:
                    raise ValueError("Theta cannot be less than 0 for Frank")
                elif theta == 0:
                    return np.multiply(U, V)
                else:
                    num = theta * (1 - np.exp(-theta)) * np.exp(-theta * (U + V))
                    den = np.power(
                        (1.0 - np.exp(-theta)) -
                        (1.0 - np.exp(-theta * U) * (1.0 - np.exp(-theta * V))), 2)
                    return num / den
            return pdf

        elif self.cname == 'gumbel':
            def pdf(U, V, theta):
                if theta < 1:
                    raise ValueError("Theta cannot be less than 1 for Gumbel")
                elif theta == 1:
                    return np.multiply(U, V)
                else:
                    cdf = Copula(U, V, theta=theta, cname='gumbel').cdf(U, V, theta)
                    a = np.power(np.multiply(U, V), -1)
                    tmp = np.power(-np.log(U), theta) + np.power(-np.log(V), theta)
                    b = np.power(tmp, -2 + 2.0 / theta)
                    c = np.power(np.multiply(np.log(U), np.log(V)), theta - 1)
                    d = 1 + (theta - 1) * np.power(tmp, -1.0 / theta)
                    return cdf * a * b * c * d
            return pdf

        else:
            raise Exception('Unsupported distribution: ' + str(self.cname))

    def _get_cdf(self):
        """Compute cdf function for given copula family
        """
        if self.cname == 'clayton':
            def cdf(U, V, theta):
                if theta < 0:
                    raise ValueError("Theta cannot be than 0 for clayton")
                elif theta == 0:
                    return np.multiply(U, V)
                else:
                    cdfs = [
                        np.power(np.power(U[i], -theta) + np.power(V[i], -theta) - 1, -1.0 / theta)
                        if U[i] > 0 else 0 for i in range(len(U))
                    ]
                    return [max(x, 0) for x in cdfs]
            return cdf

        elif self.cname == 'frank':
            def cdf(U, V, theta):
                if theta < 0:
                    raise ValueError("Theta cannot be less than 0 for Frank")
                elif theta == 0:
                    return np.multiply(U, V)
                else:
                    num = np.multiply(
                        np.exp(np.multiply(-theta, U)) - 1, np.exp(np.multiply(-theta, V)) - 1)
                    den = np.exp(-theta) - 1
                    return -1.0 / theta * np.log(1 + num / den)
            return cdf

        elif self.cname == 'gumbel':
            def cdf(U, V, theta):
                if theta < 1:
                    raise ValueError("Theta cannot be less than 1 for Gumbel")
                elif theta == 1:
                    return np.multiply(U, V)
                else:
                    h = np.power(-np.log(U), theta) + np.power(-np.log(V), theta)
                    h = -np.power(h, 1.0 / theta)
                    cdfs = np.exp(h)
                    return cdfs
            return cdf

        else:
            raise Exception('Unsupported distribution: ' + str(self.cname))

    def _get_ppf(self):
        """compute the inverse of conditional CDF C(u|v)^-1
        Args:
            y: value of C(u|v)
            v : given value of v
        """
        if self.cname == 'clayton':
            def ppf(y, v, theta):
                if theta < 0:
                    return v
                else:
                    a = np.power(y, theta / (-1 - theta))
                    b = np.power(v, theta)
                    u = np.power((a + b - 1) / b, -1 / theta)
                    return u
            return ppf
        elif self.cname == 'frank':
            def ppf(y, v, theta):
                if theta < 0:
                    return v
                else:
                    cop = Copula(1, 1, theta, 'frank', dev=True).derivative
                    u = brentq(cop, 0.0, 1.0, args=(v, theta, y))
                    return u
            return ppf
        elif self.cname == 'gumbel':
            def ppf(y, v, theta):
                if theta == 1:
                    return y
                else:
                    cop = Copula(1, 1, theta, 'gumbel', dev=True).derivative
                    u = brentq(cop, 0.0, 1.0, args=(v, theta, y))
                    return u
            return ppf
        else:
            raise Exception('Unsupported distribution: ' + str(self.cname))

    def _get_du(self):
        """Compute partial derivative of each copula cdf function
        :param theta: single parameter of the Archimedean copula
        :param cname: name of the copula function
        """
        if self.cname == 'clayton':
            def du(u, v, theta):
                if theta == 0:
                    return v
                else:
                    A = np.power(u, theta)
                    B = np.power(v, -theta) - 1
                    h = 1 + np.multiply(A, B)
                    h = np.power(h, (-1 - theta) / theta)
                    return h
            return du

        elif self.cname == 'frank':
            def du(u, v, theta, y=None):
                if theta == 0:
                    return v
                else:

                    def g(theta, z):
                        return -1 + np.exp(-np.dot(theta, z))

                    num = np.multiply(g(u, theta), g(v, theta)) + g(v, theta)
                    den = np.multiply(g(u, theta), g(v, theta)) + g(1, theta)
                    result = num / den
                    if y:
                        result = result - y
                    return result
            return du

        elif self.cname == 'gumbel':
            def du(u, v, theta, y=None):
                if theta == 1:
                    return v
                else:
                    t1 = np.power(-np.log(u), theta)
                    t2 = np.power(-np.log(v), theta)
                    p1 = np.exp(-np.power((t1 + t2), 1.0 / theta))
                    p2 = np.power(t1 + t2, -1 + 1.0 / theta)
                    p3 = np.power(-np.log(u), theta - 1)
                    result = np.divide(np.multiply(np.multiply(p1, p2), p3), u)
                    if y:
                        result = result - y
                    return result
            return du
        else:
            raise Exception('Unsupported distribution: ' + str(self.cname))

    @staticmethod
    def compute_empirical(u, v):
        """compute empirical distribution
        """
        z_left, z_right = [], []
        L, R = [], []
        N = len(u)
        base = np.linspace(0.0, 1.0, 50)
        for k in range(len(base)):
            left = sum(np.logical_and(u <= base[k], v <= base[k])) / N
            right = sum(np.logical_and(u >= base[k], v >= base[k])) / N
            if left > 0:
                z_left.append(base[k])
                L.append(left / base[k]**2)
            if right > 0:
                z_right.append(base[k])
                R.append(right / (1 - z_right[k])**2)
        return z_left, L, z_right, R

    @staticmethod
    def select_copula(U, V):
        """Select best copula function based on likelihood
        """
        clayton_c = Copula(U, V, cname='clayton')
        frank_c = Copula(U, V, cname='frank')
        gumbel_c = Copula(U, V, cname='gumbel')
        theta_c = [clayton_c.theta, frank_c.theta, gumbel_c.theta]
        if clayton_c.tau <= 0:
            bestC = 1
            paramC = frank_c.theta
            return bestC, paramC
        z_left, L, z_right, R = Copula.compute_empirical(U, V)
        left_dependence, right_dependence = [], []
        left_dependence.append(
            clayton_c.cdf(z_left, z_left, clayton_c.theta) / np.power(z_left, 2))
        left_dependence.append(frank_c.cdf(z_left, z_left, frank_c.theta) / np.power(z_left, 2))
        left_dependence.append(gumbel_c.cdf(z_left, z_left, gumbel_c.theta) / np.power(z_left, 2))

        def g(c, z):
            return np.divide(1.0 - 2 * np.asarray(z) + c, np.power(1.0 - np.asarray(z), 2))

        right_dependence.append(g(clayton_c.cdf(z_right, z_right, clayton_c.theta), z_right))
        right_dependence.append(g(frank_c.cdf(z_right, z_right, frank_c.theta), z_right))
        right_dependence.append(g(gumbel_c.cdf(z_right, z_right, gumbel_c.theta), z_right))
        # compute L2 distance from empirical distribution
        cost_L = [np.sum((L - l) ** 2) for l in left_dependence]
        cost_R = [np.sum((R - r) ** 2) for r in right_dependence]
        cost_LR = np.add(cost_L, cost_R)
        bestC = np.argmax(cost_LR)
        paramC = theta_c[bestC]
        return bestC, paramC

    def density_gaussian(self, u):
        """Compute density of gaussian copula
        """
        R = np.linalg.cholesky(self.param)
        x = scipy.stats.norm.ppf(u)
        z = np.linalg.solve(R, x.T)
        log_sqrt_det_rho = np.sum(np.log(np.diag(R)))
        y = np.exp(-0.5 * np.sum(np.power(z.T, 2) - np.power(x, 2), axis=1) - log_sqrt_det_rho)
        return y

    def _frank_help(self, alpha):
        """compute first order debye function to estimate theta
        """

        def debye(t):
            return t / (np.exp(t) - 1)

        # debye_value = quad(debye, sys.float_info.epsilon, alpha)[0] / alpha
        diff = (1 - self.tau) / 4.0 - (debye(-alpha) - 1) / alpha
        return np.power(diff, 2)
