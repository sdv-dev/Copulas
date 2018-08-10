from enum import Enum

import numpy as np
from scipy import stats


class CopulaTypes(Enum):
    CLAYTON = 0
    FRANK = 1
    GUMBEL = 2


class Bivariate(object):
    """Base class for all bivariate copulas."""

    copula_type = None
    _subclasses = []

    @classmethod
    def _get_subclasses(cls):
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass._get_subclasses())

        return subclasses

    @classmethod
    def subclasses(cls):
        if not cls._subclasses:
            cls._subclasses = cls._get_subclasses()

        return cls._subclasses

    def __new__(cls, copula_type=None):
        if not isinstance(copula_type, CopulaTypes):
            if (isinstance(copula_type, str) and copula_type.upper() in CopulaTypes.__members__):
                copula_type = CopulaTypes[copula_type.upper()]
            else:
                raise ValueError('Invalid copula type {}'.format(copula_type))

        for subclass in cls.subclasses():
            if subclass.copula_type is copula_type:
                return super(Bivariate, cls).__new__(subclass)

    def __init__(self, copula_type=None):
        """ initialize copula object """

    def fit(self, U, V):
        """ Fits a model to the data and updates the parameters """
        self.U = U
        self.V = V
        self.tau = stats.kendalltau(self.U, self.V)[0]
        self.theta = self.tau_to_theta()

    def infer(self, values):
        """ Takes in subset of values and predicts the rest """
        raise NotImplementedError

    def get_pdf(self):
        """ returns pdf of model """
        raise NotImplementedError

    def get_cdf(self):
        """ returns cdf of model """
        raise NotImplementedError

    def copula_sample(self, v, c, amount):
        raise NotImplementedError

    def sample(self, amount):
        """ returns a new data point generated from model
        v~U[0,1],v~C^-1(u|v)
        """
        if self.tau > 1 or self.tau < -1:
            raise ValueError("The range for correlation measure is [-1,1].")

        v = np.random.uniform(0, 1, amount)
        c = np.random.uniform(0, 1, amount)

        u = self.copula_sample(v, c, amount)
        U = np.column_stack((u.flatten(), v))
        return U

    def tau_to_theta(self):
        """returns theta parameter."""
        raise NotImplementedError

    @staticmethod
    def g(c, z):
        return np.divide(1.0 - 2 * np.asarray(z) + c, np.power(1.0 - np.asarray(z), 2))

    @classmethod
    def get_dependences(cls, copulas, z_left, z_right):
        left = right = []

        for copula in copulas:
            left.append(copula.get_cdf()(z_left, z_left) / np.power(z_left, 2))

        for copula in copulas:
            right.append(cls.g(copula.get_cdf()(z_right, z_right), z_right))

        return left, right

    @classmethod
    def select_copula(cls, U, V):
        """Select best copula function based on likelihood.
        :param: U An 1d array
        :param: V: An 1d array
        """
        clayton = Bivariate(CopulaTypes.CLAYTON)
        clayton.fit(U, V)

        if clayton.tau <= 0:
            frank = Bivariate(CopulaTypes.FRANK)
            frank.fit(U, V)
            selected_theta = frank.theta
            selected_copula = frank.copula_type

            return selected_copula, selected_theta

        frank = Bivariate(CopulaTypes.FRANK)
        frank.fit(U, V)
        gumbel = Bivariate(CopulaTypes.GUMBEL)
        gumbel.fit(U, V)
        candidate_copulas = [clayton, frank, gumbel]
        theta_candidates = [clayton.theta, frank.theta, gumbel.theta]

        z_left, L, z_right, R = cls.compute_empirical(U, V)
        left_dependence, right_dependence = cls.get_dependences(candidate_copulas, z_left, z_right)

        # compute L2 distance from empirical distribution
        cost_L = [np.sum((L - l) ** 2) for l in left_dependence]
        cost_R = [np.sum((R - r) ** 2) for r in right_dependence]
        cost_LR = np.add(cost_L, cost_R)
        selected_copula = np.argmax(cost_LR)
        selected_theta = theta_candidates[selected_copula]
        return selected_copula, selected_theta

    @staticmethod
    def compute_empirical(u, v):
        """compute empirical distribution"""
        z_left = z_right = []
        L = R = []
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
