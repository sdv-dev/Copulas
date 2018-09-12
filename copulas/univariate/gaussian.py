import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from copulas.univariate.base import Univariate

LOGGER = logging.getLogger(__name__)


class GaussianUnivariate(Univariate):
    """ Gaussian univariate model """

    def __init__(self):
        super().__init__()
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
        """Fits the model.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            None
        """

        if not len(X):
            raise ValueError("Can't fit with an empty dataset.")

        self.name = X.name if isinstance(X, (pd.Series, pd.DataFrame)) else None
        self.mean = np.mean(X)
        self.std = np.std(X) or 0.001

    def probability_density(self, X):
        """Computes probability density.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray
        """
        return norm.pdf(X, loc=self.mean, scale=self.std)

    def pdf(self, X):
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Cumulative density function for gaussian distribution.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray: Cumulative density for X.
        """

        return norm.cdf(X, loc=self.mean, scale=self.std)

    def cdf(self, X):
        return self.cumulative_distribution(X)

    def percent_point(self, U):
        """Given a cumulated density, returns a value in original space.

        Arguments:
            U: `np.ndarray` of shape (n, 1) and values in [0,1]

        Returns:
            `np.ndarray`: Estimated values in original space.
        """
        return norm.ppf(U, loc=self.mean, scale=self.std)

    def ppf(self, U):
        return self.percent_point(U)

    def sample(self, num_samples=1):
        """Returns new data point based on model.

        Arguments:
            n_samples: `int`

        Returns:
            np.ndarray: Generated samples
        """
        return np.random.normal(self.mean, self.std, num_samples)

    def to_dict(self):
        return {
            'mean': self.mean,
            'std': self.std
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Set attributes with provided values."""
        instance = cls()
        instance.mean = copula_dict['mean']
        instance.std = copula_dict['std']
        return instance
