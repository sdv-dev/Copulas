import logging

import numpy as np
from copulas.univariate.UnivariateDistrib import UnivariateDistrib
from scipy.stats import norm

LOGGER = logging.getLogger(__name__)


class GaussianUnivariate(UnivariateDistrib):
    """ Gaussian univariate model """

    def __init__(self):
        super(GaussianUnivariate, self).__init__()
        self.column = None
        self.mean = 0
        self.std = 1
        self.min = -np.inf
        self.max = np.inf

    def fit(self, column):
        LOGGER.debug('Distribution Type: Gaussian')
        self.column = column
        LOGGER.debug('Variable name: ', self.column.name)
        self.mean = np.mean(column)
        LOGGER.debug('mean = ', self.mean)
        self.std = np.std(column)
        LOGGER.debug('standard deviation = ', self.std)
        self.max = max(column)
        LOGGER.debug('max = ', self.max)
        self.min = min(column)
        LOGGER.debug('min = ', self.min)

    def get_pdf(self, x):
        return norm.pdf(x, loc=self.mean, scale=self.std)

    def get_cdf(self, x):
        return norm.cdf(x, loc=self.mean, scale=self.std)

    # def _calculate_cdf(self):
    #   def cdf(data):
    #       u = []
    #       for y in data:
    #           ui = self.pdf.integrate_box_1d(-np.inf, y)
    #           u.append(ui)
    #       u = np.asarray(u)
    #       return u
    #   return cdf

    def inverse_cdf(self, u):
        """ given a cdf value, returns a value in original space """
        return norm.ppf(u, loc=self.mean, scale=self.std)

    def sample(self, num_samples=1):
        """ returns new data point based on model """
        return np.random.normal(self.mean, self.std, num_samples)
