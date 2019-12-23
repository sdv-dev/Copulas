import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from copulas import check_valid_values, random_state, store_args
from copulas.univariate.base import Univariate

LOGGER = logging.getLogger(__name__)


class GaussianUnivariate(Univariate):
    """Gaussian univariate model."""

    fitted = False

    @store_args
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None
        self.mean = None
        self.std = None

    def __str__(self):
        details = [self.name, self.mean, self.std]
        return (
            'Distribution Type: Gaussian\n'
            'Variable name: {}\n'
            'Mean: {}\n'
            'Standard deviation: {}'.format(*details)
        )

    @check_valid_values
    def fit(self, X):
        """Fit the model.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            None
        """
        if isinstance(X, (pd.Series, pd.DataFrame)):
            self.name = X.name

        self.constant_value = self._get_constant_value(X)

        if self.constant_value is None:
            self.mean = np.mean(X)
            self.std = np.std(X)

        else:
            self._replace_constant_methods()

        self.fitted = True

    def probability_density(self, X):
        """Compute probability density.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray
        """
        self.check_fit()
        return norm.pdf(X, loc=self.mean, scale=self.std)

    def cumulative_distribution(self, X):
        """Cumulative distribution function for gaussian distribution.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray: Cumulative density for X.
        """
        self.check_fit()
        return norm.cdf(X, loc=self.mean, scale=self.std)

    def percent_point(self, U):
        """Given a cumulated distribution value, returns a value in original space.

        Arguments:
            U: `np.ndarray` of shape (n, 1) and values in [0,1]

        Returns:
            `np.ndarray`: Estimated values in original space.
        """
        self.check_fit()
        return norm.ppf(U, loc=self.mean, scale=self.std)

    @random_state
    def sample(self, num_samples=1):
        """Returns new data point based on model.

        Arguments:
            n_samples: `int`

        Returns:
            np.ndarray: Generated samples
        """
        self.check_fit()
        return np.random.normal(self.mean, self.std, num_samples)

    def _fit_params(self):
        if self.constant_value is not None:
            return {
                'mean': self.constant_value,
                'std': 0
            }

        return {
            'mean': self.mean,
            'std': self.std,
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Set attributes with provided values."""
        instance = cls()
        instance.fitted = copula_dict['fitted']

        if not instance.fitted:
            return instance

        std = copula_dict['std']
        if std == 0:
            instance.constant_value = copula_dict['mean']

        else:
            instance.mean = copula_dict['mean']
            instance.std = std

        return instance
