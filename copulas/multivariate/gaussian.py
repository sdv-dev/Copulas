import logging
import sys

import numpy as np
import pandas as pd
from scipy import stats

from copulas import (
    EPSILON, check_valid_values, get_instance, get_qualified_name, random_state, store_args)
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

        self.columns = []
        self.univariates = []
        self.covariance = None
        self.distribution = distribution

    def __str__(self):
        univariates = [
            '\n{}\n==============\n{}'.format(key, value)
            for key, value in zip(self.columns, self.univariates)
        ]

        covariance = (
            '\n\nCovariance:\n{}'.format(self.covariance)
        )
        return '\n'.join(univariates) + covariance

    def _transform_to_normal(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = [X]

            X = pd.DataFrame(X, columns=self.columns)

        U = list()
        for column_name, univariate in zip(self.columns, self.univariates):
            column = X[column_name]
            U.append(univariate.cdf(column).clip(EPSILON, 1 - EPSILON))

        return stats.norm.ppf(np.column_stack(U))

    def _get_covariance(self, X):
        """Compute covariance matrix with transformed data.

        Args:
            X: `numpy.ndarray` or `pandas.DataFrame`.

        Returns:
            np.ndarray

        """
        result = self._transform_to_normal(X)
        covariance = pd.DataFrame(data=result).cov().values
        # If singular, add some noise to the diagonal
        if np.linalg.cond(covariance) > 1.0 / sys.float_info.epsilon:
            covariance = covariance + np.identity(covariance.shape[0]) * EPSILON
        return covariance

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

            univariate = get_instance(distribution)
            univariate.fit(column)

            self.columns.append(column_name)
            self.univariates.append(univariate)

        self.covariance = self._get_covariance(X)
        self.fitted = True

    def probability_density(self, X):
        """Evaluate the probability density function at `X`.

        Args:
            X (numpy.ndarray or pandas.DataFrame):
                Points at which the probability density function
                will be evaluated.

        Returns:
            numpy.ndarray:
                Probability density for the input values.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)
        return stats.multivariate_normal.pdf(transformed, cov=self.covariance)

    def cumulative_distribution(self, X):
        """Evaluate the cumulative distribution function at `X`.

        Args:
            X (numpy.ndarray or pandas.DataFrame):
                Points at which the cumulative distribution function
                will be evaluated.

        Returns:
            numpy.ndarray:
                Cumulative Probability
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)
        return stats.multivariate_normal.cdf(transformed, cov=self.covariance)

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

        for i, (column_name, univariate) in enumerate(zip(self.columns, self.univariates)):
            cdf = stats.norm.cdf(samples[:, i])
            res[column_name] = univariate.percent_point(cdf)

        return pd.DataFrame(data=res)

    def to_dict(self):
        univariates = [univariate.to_dict() for univariate in self.univariates]
        distribution = self.distribution
        if isinstance(self.distribution, dict):
            distribution = {}
            for k, v in self.distribution.items():
                distribution[k] = v.to_dict()

        return {
            'covariance': self.covariance.tolist(),
            'univariates': univariates,
            'columns': self.columns,
            'type': get_qualified_name(self),
            'fitted': self.fitted,
            'distribution': distribution
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Set attributes with provided values."""
        instance = cls()
        instance.univariates = []
        instance.columns = copula_dict['columns']

        for parameters in copula_dict['univariates']:
            instance.univariates.append(Univariate.from_dict(parameters))

        instance.covariance = np.array(copula_dict['covariance'])
        instance.fitted = copula_dict['fitted']
        instance.distribution = copula_dict['distribution']
        if isinstance(instance.distribution, dict):
            for k, v in instance.distribution.items():
                instance.distribution[k] = Univariate.from_dict(v)
        return instance
