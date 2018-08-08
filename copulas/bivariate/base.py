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
