import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from copulas.univariate.base import Univariate

LOGGER = logging.getLogger(__name__)


class GaussianUnivariate(Univariate):
    """ Gaussian univariate model """

    def __init__(self):
        super(GaussianUnivariate, self).__init__()
        self.name = None
        self.mean = 0
        self.std = 1

    def __str__(self):
        details = [self.name, self.mean, self.std]
        return (
            'Distribution Type: Gaussian\n'
            'Variable name: {}\n'
            'Mean: {}\n'
            'Standard deviation: {}'.format(*details)
        )

    def fit(self, X):
        if not len(X):
            raise ValueError("Can't fit with an empty dataset.")

        self.name = X.name if isinstance(X, pd.Series) else None
        self.mean = np.mean(X)
        std = np.std(X)

        # check for column with all the same vals
        if std == 0:
            self.std = 0.001

        else:
            self.std = std

    def probability_density(self, X):
        return norm.pdf(X, loc=self.mean, scale=self.std)

    def cumulative_density(self, X):
        """Cumulative density function for gaussian distribution."""
        # check to make sure dtype is not object
        if X.dtype == 'object':
            X = X.astype('float64')
        return norm.cdf(X, loc=self.mean, scale=self.std)

    def percent_point(self, u):
        """Returns a value in original space given a cdf."""
        return norm.ppf(u, loc=self.mean, scale=self.std)

    def sample(self, num_samples=1):
        """Returns new data point based on model.

        Argument:
            n_samples: `int`

        Returns:
            np.ndarray: Generated samples
        """
        return np.random.normal(self.mean, self.std, num_samples)
