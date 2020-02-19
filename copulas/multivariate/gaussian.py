import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats

from copulas import check_valid_values, get_instance, get_qualified_name, random_state, store_args
from copulas.multivariate.base import Multivariate
from copulas.univariate import Univariate

LOGGER = logging.getLogger(__name__)
DEFAULT_DISTRIBUTION = 'copulas.univariate.Univariate'


class GaussianMultivariate(Multivariate):
    """Class for a multivariate distribution that uses the Gaussian copula.

    Args:
        distribution (str or dict):
            Fully qualified name of the class to be used for modeling the marginal
            distributions or a dictionary mapping column names to the fully qualified
            distribution names.
    """

    @store_args
    def __init__(self, distribution=DEFAULT_DISTRIBUTION, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.distribs = OrderedDict()
        self.covariance = None
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

    def _get_covariance(self, X):
        """Compute covariance matrix with transformed data.

        Args:
            X: `numpy.ndarray` or `pandas.DataFrame`.

        Returns:
            np.ndarray

        """
        result = self._transform_to_normal(X)
        return pd.DataFrame(data=result).cov().values

    def _transform_to_normal(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        U = X.copy()
        for (column_name, column), distrib in zip(X.items(), self.distribs.values()):
            if distrib.constant_value is not None:
                U[column_name] = np.ones(column.shape) / 2.0
            else:
                U[column_name] = distrib.cdf(column)
        return stats.norm.ppf(U)

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
            if isinstance(self.distribution, dict):
                distribution = self.distribution.get(column_name, DEFAULT_DISTRIBUTION)
            else:
                distribution = self.distribution

            distribution_instance = get_instance(distribution)
            distribution_instance.fit(column)

            self.distribs[column_name] = distribution_instance

        self.covariance = self._get_covariance(X)
        self.fitted = True

    def probability_density(self, x):
        """Evaluate the probability density function at `x` after transforming it
        into the appropriate distribution.

        Args:
            x: `numpy.ndarray` or `pandas.DataFrame`

        Returns:
            np.array: Probability density for the input values.
        """
        self.check_fit()
        transformed = self._transform_to_normal([x])
        pdf = stats.multivariate_normal.pdf(transformed, cov=self.covariance)
        return pdf

    def cumulative_distribution(self, x):
        """Evaluate the cumulative distribution function at `x` after transforming
        it into the appropriate distribution.

        Args:
            X: `numpy.ndarray` or `pandas.DataFrame`

        Returns:
            np.array: cumulative probability
        """
        self.check_fit()
        transformed = self._transform_to_normal([x])
        cdf = stats.multivariate_normal.cdf(transformed, cov=self.covariance)
        return cdf

    @random_state
    def sample(self, num_rows=1):
        """Creates synthetic values statistically similar to the original dataset.

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
