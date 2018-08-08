import numpy as np

from copulas.bivariate.base import Bivariate, CopulaTypes


class Clayton(Bivariate):
    """ Class for clayton copula model """

    copula_type = CopulaTypes.CLAYTON

    def get_generator(self):
        """Return the generator function.
        """
        def generator(theta, t):
            return 1.0 / theta * (np.power(t, -theta) - 1)
        return generator

    def get_pdf(self):
        """compute density function for given copula family
        """
        def pdf(U, V):
            if self.theta < 0:
                raise ValueError("Theta cannot be less or equal than 0 for clayton")
            elif self.theta == 0:
                return np.multiply(U, V)
            else:
                a = (self.theta + 1) * np.power(np.multiply(U, V), -(self.theta + 1))
                b = np.power(U, -self.theta) + np.power(V, -self.theta) - 1
                c = -(2 * self.theta + 1) / self.theta
                density = a * np.power(b, c)
                return density
        return pdf

    def get_cdf(self):
        """Compute cdf function for given copula family
        """
        def cdf(U, V):
            if self.theta < 0:
                raise ValueError("Theta cannot be less or equal than 0 for clayton")
            elif self.theta == 0:
                return np.multiply(U, V)
            elif U == 0 or V == 0:
                return 0
            elif type(U) in (int, float):
                value = np.power(
                    np.power(U, -self.theta) + np.power(V, -self.theta) - 1,
                    -1.0 / self.theta)
                return value
            else:
                cdfs = [
                    np.power(
                        np.power(U[i], -self.theta) + np.power(V[i], -self.theta) - 1,
                        -1.0 / self.theta
                    )
                    if U[i] > 0 else 0 for i in range(len(U))
                ]
                return [max(x, 0) for x in cdfs]
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
                a = np.power(y, theta / (-1 - theta))
                b = np.power(v, theta)
                u = np.power((a + b - 1) / b, -1 / theta)
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
                A = np.power(u, theta)
                B = np.power(v, -theta) - 1
                h = 1 + np.multiply(A, B)
                h = np.power(h, (-1 - theta) / theta)
                h = h - y
                return h
        return du

    def tau_to_theta(self):
        if self.tau == 1:
            theta = 10000
        else:
            theta = 2 * self.tau / (1 - self.tau)
        return theta

    def copula_sample(self, v, c, amount):
        return self.get_ppf()(c, v, self.theta)
