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
        """Fit the model.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            None
        """

        if not len(X):
            raise ValueError("Can't fit with an empty dataset.")

        if isinstance(X, (pd.Series, pd.DataFrame)):
            self.name = X.name
        else:
            self.name = None

        self.mean = np.mean(X)
        self.std = np.std(X) or 0.001

    def probability_density(self, X):
        """Compute probability density.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray
        """
        return norm.pdf(X, loc=self.mean, scale=self.std)

    def cumulative_distribution(self, X):
        """Cumulative distribution function for gaussian distribution.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray: Cumulative density for X.
        """

        return norm.cdf(X, loc=self.mean, scale=self.std)

    def percent_point(self, U):
        """Given a cumulated distribution value, returns a value in original space.

        Arguments:
            U: `np.ndarray` of shape (n, 1) and values in [0,1]

        Returns:
            `np.ndarray`: Estimated values in original space.
        """
        return norm.ppf(U, loc=self.mean, scale=self.std)

    def sample(self, num_samples=1, seed=None):
        """Returns new data point based on model.

        Arguments:
            n_samples: `int`
            
            seed: `int` or None, the seed for the random numbers generator.

        Returns:
            np.ndarray: Generated samples
        """
        s = np.random.get_state()
        
        np.random.seed(seed)
        
        sample = np.random.normal(self.mean, self.std, num_samples)
        
        np.random.set_state(s)
        
        return sample

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
