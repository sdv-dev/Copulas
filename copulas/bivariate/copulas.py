import logging

import numpy as np
import scipy.stats
from scipy.optimize import fmin, brentq, fminbound

LOGGER = logging.getLogger(__name__)

eps = np.finfo(np.float32).eps

class CopulaException(Exception):
    pass


class Copula(object):
    def __init__(self, cname):
        """Instantiates an instance of the copula object from a pandas dataframe
        :param cname: the choice of copulas, can be 'clayton','gumbel','frank','gaussian'
        """
        self.cname = cname

    def fit(self,U,V):
        """ Fit a copula object.
        """
        self.U = U
        self.V = V
        self.tau = stats.kendalltau(self.U, self.V)[0]
        self.theta = Copula.tau_to_theta(self.cname,self.tau)


    def get_params(self):
        return {'tau':self.tau,'theta':self.theta}

    def set_params(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)

    def get_generator(self):
        """Return the generator function.
        """
        if self.cname == 'clayton':
            def generator(theta, t):
                return 1.0/theta*(np.power(t,-theta)-1)
            return generator
        elif self.cname == 'frank':
            def generator(theta, t):
                a = (np.exp(-theta*t)-1)/(np.exp(-theta)-1)
                return -np.log(a)
            return generator
        elif self.cname == 'gumebl':
            def generator(theta, t):
                return np.power(-np.log(t),theta)
            return generator

    def get_pdf(self):
        """compute density function for given copula family
        """
        if self.cname == 'clayton':
            def pdf(U, V):
                if self.theta < 0:
                    raise ValueError("Theta cannot be than 0 for clayton")
                elif self.theta == 0:
                    return np.multiply(U, V)
                else:
                    a = (self.theta + 1) * np.power(np.multiply(U, V), -(self.theta + 1))
                    b = np.power(U, -self.theta) + np.power(V, -self.theta) - 1
                    c = -(2 * self.theta + 1) / self.theta
                    density = a * np.power(b, c)
                    return density
            return pdf

        elif self.cname == 'frank':
            def pdf(U, V):
                if self.theta < 0:
                    raise ValueError("Theta cannot be less than 0 for Frank")
                elif self.theta == 0:
                    return np.multiply(U, V)
                else:
                    num = self.theta * (1 - np.exp(-self.theta)) * np.exp(-self.theta * (U + V))
                    den = np.power(
                        (1.0 - np.exp(-self.theta)) -
                        (1.0 - np.exp(-self.theta * U) * (1.0 - np.exp(-self.theta * V))), 2)
                    return num / den
            return pdf

        elif self.cname == 'gumbel':
            def pdf(U, V):
                if self.theta < 1:
                    raise ValueError("Theta cannot be less than 1 for Gumbel")
                elif self.theta == 1:
                    return np.multiply(U, V)
                else:
                    cop = Copula('gumbel').fit(U, V)
                    cdf=cop.get_cdf()
                    a = np.power(np.multiply(U, V), -1)
                    tmp = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
                    b = np.power(tmp, -2 + 2.0 / self.theta)
                    c = np.power(np.multiply(np.log(U), np.log(V)), self.theta - 1)
                    d = 1 + (self.theta - 1) * np.power(tmp, -1.0 / self.theta)
                    return cdf(U, V) * a * b * c * d
            return pdf

        else:
            raise Exception('Unsupported distribution: ' + str(self.cname))

    def get_cdf(self):
        """Compute cdf function for given copula family
        """
        if self.cname == 'clayton':
            def cdf(U, V):
                if self.theta < 0:
                    raise ValueError("Theta cannot be than 0 for clayton")
                elif self.theta == 0:
                    return np.multiply(U, V)
                else:
                    cdfs = [
                        np.power(np.power(U[i], -self.theta) + np.power(V[i], -self.theta) - 1, -1.0 / self.theta)
                        if U[i] > 0 else 0 for i in range(len(U))
                    ]
                    return [max(x, 0) for x in cdfs]
            return cdf

        elif self.cname == 'frank':
            def cdf(U, V):
                if self.theta < 0:
                    raise ValueError("Theta cannot be less than 0 for Frank")
                elif self.theta == 0:
                    return np.multiply(U, V)
                else:
                    num = np.multiply(
                        np.exp(np.multiply(-self.theta, U)) - 1, np.exp(np.multiply(-self.theta, V)) - 1)
                    den = np.exp(-self.theta) - 1
                    return -1.0 / self.theta * np.log(1 + num / den)
            return cdf

        elif self.cname == 'gumbel':
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

        else:
            raise Exception('Unsupported distribution: ' + str(self.cname))

    def get_ppf(self):
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
                    dev = Copula('frank').get_h_function()
                    u= fminbound(dev,eps,1.0,args=(v,theta,y))
                    return u
            return ppf
        elif self.cname == 'gumbel':
            def ppf(y, v, theta):
                if theta == 1:
                    return y
                else:
                    dev = Copula('gumbel').get_h_function()
                    u = fminbound(dev,eps,1.0,args=(v,theta,y))
                    return u
            return ppf
        else:
            raise Exception('Unsupported distribution: ' + str(self.cname))

    def get_h_function(self):
        """Compute partial derivative C(u|v) of each copula cdf function
        :param theta: single parameter of the Archimedean copula
        :param cname: name of the copula function
        """
        if self.cname == 'clayton':
            def du(u, v, theta, y=0):
                if theta == 0:
                    return v
                else:
                    A = np.power(u, theta)
                    B = np.power(v, -theta) - 1
                    h = 1 + np.multiply(A, B)
                    h = np.power(h, (-1 - theta) / theta)
                    h = h-y
                    return h
            return du

        elif self.cname == 'frank':
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

        elif self.cname == 'gumbel':
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
        else:
            raise Exception('Unsupported distribution: ' + str(self.cname))

    @staticmethod
    def tau_to_theta(cname, tau):
        if cname == 'clayton':
            if tau == 1:
                theta = 10000
            else:
                theta = 2*tau/(1-tau)

        elif cname == 'frank':
            theta = -fmin(Copula._frank_help, -5, args=(tau,), disp=False)[0]

        elif cname == 'gumbel':
            if tau == 1:
                theta = 10000
            else:
                theta = 1/(1-tau)
        return theta

    @staticmethod
    def sampling(cname,tau,n_sample):
        """sampling from bivariate copula given tau
        v~U[0,1],v~C^-1(u|v)
        """
        eps = np.finfo(np.float32).eps
        if tau>1 or tau<-1:
            raise ValueError("The range for correlation measure is [-1,1].")
        v = np.random.uniform(0,1,n_sample)
        c = np.random.uniform(0,1,n_sample)
        cop = Copula(cname)
        theta = Copula.tau_to_theta(cname,tau)
        print(theta)
        ppf = cop.get_ppf()
        if cname == 'clayton':
            u = ppf(c,v,theta)
        elif cname == 'frank':
            u = np.empty([1,n_sample])
            for i in range(len(v)):
                u[0,i] = ppf(c[i],v[i],theta)
            print(u)
        elif cname == 'gumbel':
            u = np.empty([1,n_sample])
            for i in range(len(v)):
                u[0,i] = ppf(c[i],v[i],theta)
            print(u)
        else:
            u = np.random.uniform(0,1,n_sample)
        U = np.column_stack((u.flatten(),v))
        return U

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

    # def density_gaussian(self, u):
    #     """Compute density of gaussian copula
    #     """
    #     R = np.linalg.cholesky(self.param)
    #     x = scipy.stats.norm.ppf(u)
    #     z = np.linalg.solve(R, x.T)
    #     log_sqrt_det_rho = np.sum(np.log(np.diag(R)))
    #     y = np.exp(-0.5 * np.sum(np.power(z.T, 2) - np.power(x, 2), axis=1) - log_sqrt_det_rho)
    #     return y

    @staticmethod
    def _frank_help(alpha, tau):
        """compute first order debye function to estimate theta
        """

        def debye(t):
            return t / (np.exp(t) - 1)

        # debye_value = quad(debye, sys.float_info.epsilon, alpha)[0] / alpha
        diff = (1 - self.tau) / 4.0 - (debye(-alpha) - 1) / alpha
        return np.power(diff, 2)
