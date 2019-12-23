import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import integrate, stats

from copulas import (
    EPSILON, check_valid_values, get_instance, get_qualified_name, random_state, store_args)
from copulas.multivariate.base import Multivariate
from copulas.univariate import Univariate

LOGGER = logging.getLogger(__name__)
DEFAULT_DISTRIBUTION = 'copulas.univariate.gaussian.GaussianUnivariate'


class GaussianMultivariate(Multivariate):
    """Class for a gaussian copula model.

    Args:
        distribution (str): Full qualified name of the class to be used as distribution.
    """

    @store_args
    def __init__(self, distribution=DEFAULT_DISTRIBUTION, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.distribs = OrderedDict()
        self.covariance = None
        self.means = None
        self.distribution = distribution

    def __str__(self):
        distribs = [
            '\n{}\n==============\n{}'.format(key, value)
            for key, value in self.distribs.items()
        ]

        covariance = (
            '\n\nCovariance:\n{}'.format(self.covariance)
        )
        return '\n'.join(distribs) + covariance

    def get_lower_bound(self):
        """Compute the lower bound to integrate cumulative density.

        Returns:
            float: lower bound for cumulative density integral.
        """
        lower_bounds = []

        for distribution in self.distribs.values():
            lower_bound = distribution.percent_point(distribution.mean / 10000)
            if not pd.isnull(lower_bound):
                lower_bounds.append(lower_bound)

        return min(lower_bounds)

    def _get_covariance(self, X):
        """Compute covariance matrix with transformed data.

        Args:
            X: `numpy.ndarray` or `pandas.DataFrame`.

        Returns:
            np.ndarray

        """
        result = pd.DataFrame(index=range(len(X)))
        for column_name, column in X.items():
            distrib = self.distribs[column_name]

            # get original distrib's cdf of the column
            cdf = distrib.cumulative_distribution(column)

            if distrib.constant_value is not None:
                # This is to avoid np.inf in the case the column is constant.
                cdf = np.ones(column.shape) - EPSILON

            # get inverse cdf using standard normal
            result[column_name] = stats.norm.ppf(cdf)

        # remove any rows that have infinite values
        result = result[~result.isin([np.inf, -np.inf]).any(axis=1)]
        return pd.DataFrame(data=result).cov().values

    @check_valid_values
    def fit(self, X):
        """Compute the distribution for each variable and then its covariance matrix.

        Args:
            X(numpy.ndarray or pandas.DataFrame): Data to model.

        Returns:
            None
        """
        LOGGER.debug('Fitting Gaussian Copula')

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        for column_name, column in X.items():
            self.distribs[column_name] = get_instance(self.distribution)
            self.distribs[column_name].fit(column)

        self.covariance = self._get_covariance(X)
        self.fitted = True

    def probability_density(self, X):
        """Compute probability density function for given copula family.

        Args:
            X: `numpy.ndarray` or `pandas.DataFrame`

        Returns:
            np.array: Probability density for the input values.
        """
        self.check_fit()

        # make cov positive semi-definite
        covariance = self.covariance * np.identity(self.covariance.shape[0])
        return stats.multivariate_normal.pdf(X, cov=covariance)

    def cumulative_distribution(self, X):
        """Computes the cumulative distribution function for the copula

        Args:
            X: `numpy.ndarray` or `pandas.DataFrame`

        Returns:
            np.array: cumulative probability
        """
        self.check_fit()

        # Wrapper for pdf to accept vector as args
        def func(*args):
            return self.probability_density(list(args))

        # Lower bound for integral, to split significant part from tail
        lower_bound = self.get_lower_bound()

        ranges = [[lower_bound, val] for val in X]
        return integrate.nquad(func, ranges)[0]

    @random_state
    def sample(self, num_rows=1):
        """Creates sintentic values stadistically similar to the original dataset.

        Args:
            num_rows: `int` amount of samples to generate.

        Returns:
            np.ndarray: Sampled data.

        """
        self.check_fit()

        res = {}
        means = np.zeros(self.covariance.shape[0])
        size = (num_rows,)

        clean_cov = np.nan_to_num(self.covariance)
        samples = np.random.multivariate_normal(means, clean_cov, size=size)

        for i, (label, distrib) in enumerate(self.distribs.items()):
            cdf = stats.norm.cdf(samples[:, i])
            res[label] = distrib.percent_point(cdf)

        return pd.DataFrame(data=res)

    def to_dict(self):
        distributions = {
            name: distribution.to_dict() for name, distribution in self.distribs.items()
        }

        return {
            'covariance': self.covariance.tolist(),
            'distribs': distributions,
            'type': get_qualified_name(self),
            'fitted': self.fitted,
            'distribution': self.distribution
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Set attributes with provided values."""
        instance = cls()
        instance.distribs = {}

        for name, parameters in copula_dict['distribs'].items():
            instance.distribs[name] = Univariate.from_dict(parameters)

        instance.covariance = np.array(copula_dict['covariance'])
        instance.fitted = copula_dict['fitted']
        instance.distribution = copula_dict['distribution']
        return instance
