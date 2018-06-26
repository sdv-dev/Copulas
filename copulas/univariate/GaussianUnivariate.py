import logging

import numpy as np
from scipy.stats import norm

from copulas.univariate.UnivariateDistrib import UnivariateDistrib

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

    def __str__(self):
        details = [self.column.name, self.mean, self.std, self.max, self.min]
        return (
            'Distribution Type: Gaussian\n'
            'Variable name: {}\n'
            'Mean: {}\n'
            'Standard deviation: {}\n'
            'Max: {}\n'
            'Min: {}'.format(*details)
        )

    def fit(self, column):
        self.column = column
        self.mean = np.mean(column)
        std = np.std(column)
        # check for column with all the same vals
        if std == 0:
            self.std = 0.001
        else:
            self.std = std
        self.max = max(column)
        self.min = min(column)

    def get_pdf(self, x):
        return norm.pdf(x, loc=self.mean, scale=self.std)

    def get_cdf(self, x):
        # check to make sure dtype is not object
        if x.dtype == 'object':
            x = x.astype('float64')
        return norm.cdf(x, loc=self.mean, scale=self.std)

    def inverse_cdf(self, u):
        """ given a cdf value, returns a value in original space """
        return norm.ppf(u, loc=self.mean, scale=self.std)

    def sample(self, num_samples=1):
        """ returns new data point based on model """
        return np.random.normal(self.mean, self.std, num_samples)
