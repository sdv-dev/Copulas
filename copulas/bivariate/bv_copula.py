import numpy as np
from scipy import stats

from copulas.bivariate import clayton
from copulas.bivariate import frank
from copulas.bivariate import gumbel


class BVCopula(object):
    """ Abstract class for a bivariate copula object """

    def __init__(self, name):
        """ initialize copula object """
        self.name = name
        if self.name == 'Clayton':
            self.copula = clayton.Clayton()
        elif self.name == 'Frank':
            self.copula = frank.Frank()
        elif self.name == 'Gumbel':
            self.copula = gumbel.Gumbel()
        else:
            raise Exception('Unsupported distribution: ' + str(self.name))

    def fit(self, U, V):
        """ Fits a model to the data and updates the parameters """
        """ Fit a copula object.
        """
        self.copula.fit(U, V)
        self.U = U
        self.V = V
        self.tau = stats.kendalltau(self.U, self.V)[0]
        self.theta = self.copula.tau_to_theta()

    def get_params(self):
        return {'tau': self.tau, 'theta': self.theta}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def infer(self, values):
        """ Takes in subset of values and predicts the rest """
        raise NotImplementedError

    def get_generator(self, theta, t):
        """ return the generator function """
        return self.copula.get_generator()(theta, t)

    def get_pdf(self, U, V):
        """ returns pdf of model """
        return self.copula.get_pdf()(U, V)

    def get_cdf(self, U, V):
        """ returns cdf of model """
        return self.copula.get_cdf()(U, V)

    def get_ppf(self, y, v, theta):
        """ returns ppf of model """
        return self.copula.get_ppf()(y, v, theta)

    def get_h_function(self, u, v, theta, y=0):
        """ returns h function value of model """
        return self.copula.get_h_function()(u, v, theta, y)

    @staticmethod
    def sample(cname, tau, n_sample):
        """ returns a new data point generated from model
        v~U[0,1],v~C^-1(u|v)
        """
        if tau > 1 or tau < -1:
            raise ValueError("The range for correlation measure is [-1,1].")
        v = np.random.uniform(0, 1, n_sample)
        c = np.random.uniform(0, 1, n_sample)
        cop = BVCopula(cname)
        theta = cop.tau_to_theta(cname, tau)
        ppf = cop.get_ppf()
        if cname == 'clayton':
            u = ppf(c, v, theta)
        elif cname == 'frank':
            u = np.empty([1, n_sample])
            for i in range(len(v)):
                u[0, i] = ppf(c[i], v[i], theta)
        elif cname == 'gumbel':
            u = np.empty([1, n_sample])
            for i in range(len(v)):
                u[0, i] = ppf(c[i], v[i], theta)
        else:
            u = np.random.uniform(0, 1, n_sample)
        U = np.column_stack((u.flatten(), v))
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
        clayton_c = BVCopula('clayton')
        clayton_c.fit(U, V)
        frank_c = BVCopula('frank')
        frank_c.fit(U, V)
        gumbel_c = BVCopula('gumbel')
        gumbel_c.fit(U, V)
        theta_c = [clayton_c.theta, frank_c.theta, gumbel_c.theta]
        if clayton_c.tau <= 0:
            bestC = 1
            paramC = frank_c.theta
            return bestC, paramC
        z_left, L, z_right, R = BVCopula.compute_empirical(U, V)
        left_dependence, right_dependence = [], []
        left_dependence.append(
            clayton_c.get_cdf()(z_left, z_left) / np.power(z_left, 2))
        left_dependence.append(
            frank_c.get_cdf()(z_left, z_left) / np.power(z_left, 2))
        left_dependence.append(
            gumbel_c.get_cdf()(z_left, z_left) / np.power(z_left, 2))

        def g(c, z):
            return np.divide(1.0 - 2 * np.asarray(z) + c, np.power(1.0 - np.asarray(z), 2))

        right_dependence.append(g(clayton_c.get_cdf()(z_right, z_right), z_right))
        right_dependence.append(g(frank_c.get_cdf()(z_right, z_right), z_right))
        right_dependence.append(g(gumbel_c.get_cdf()(z_right, z_right), z_right))
        # compute L2 distance from empirical distribution
        cost_L = [np.sum((L - l) ** 2) for l in left_dependence]
        cost_R = [np.sum((R - r) ** 2) for r in right_dependence]
        cost_LR = np.add(cost_L, cost_R)
        print(left_dependence)
        print(right_dependence)
        bestC = np.argmax(cost_LR)
        paramC = theta_c[bestC]
        return bestC, paramC
