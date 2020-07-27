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
DEFAULT_DISTRIBUTION = Univariate


class GaussianMultivariate(Multivariate):
    """Class for a multivariate distribution that uses the Gaussian copula.

    Args:
        distribution (str or dict):
            Fully qualified name of the class to be used for modeling the marginal
            distributions or a dictionary mapping column names to the fully qualified
            distribution names.
    """

    covariance = None
    columns = None
    univariates = None

    @store_args
    def __init__(self, distribution=DEFAULT_DISTRIBUTION, random_seed=None):
        self.random_seed = random_seed
        self.distribution = distribution

    def __repr__(self):
        if self.distribution == DEFAULT_DISTRIBUTION:
            distribution = ''
        elif isinstance(self.distribution, type):
            distribution = 'distribution="{}"'.format(self.distribution.__name__)
        else:
            distribution = 'distribution="{}"'.format(self.distribution)

        return 'GaussianMultivariate({})'.format(distribution)

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
            U.append(univariate.cdf(column.values).clip(EPSILON, 1 - EPSILON))

        return stats.norm.ppf(np.column_stack(U))

    def _get_covariance(self, X):
        """Compute covariance matrix with transformed data.

        Args:
            X (numpy.ndarray):
                Data for which the covariance needs to be computed.

        Returns:
            numpy.ndarray:
                computed covariance matrix.
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

        Arguments:
            X (pandas.DataFrame):
                Values of the random variables.
        """
        LOGGER.info('Fitting %s', self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        columns = []
        univariates = []
        for column_name, column in X.items():
            if isinstance(self.distribution, dict):
                distribution = self.distribution.get(column_name, DEFAULT_DISTRIBUTION)
            else:
                distribution = self.distribution

            LOGGER.debug('Fitting column %s to %s', column_name, distribution)

            univariate = get_instance(distribution)
            univariate.fit(column)

            columns.append(column_name)
            univariates.append(univariate)

        self.columns = columns
        self.univariates = univariates

        LOGGER.debug('Computing covariance')
        self.covariance = self._get_covariance(X)
        self.fitted = True

        LOGGER.debug('GaussianMultivariate fitted successfully')

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the probability density will be computed.

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)
        return stats.multivariate_normal.pdf(transformed, cov=self.covariance)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the cumulative distribution will be computed.

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)
        return stats.multivariate_normal.cdf(transformed, cov=self.covariance)

    @random_state
    def sample(self, num_rows=1):
        """Sample values from this model.

        Argument:
            num_rows (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, *) with values randomly
                sampled from this model distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
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
        """Return a `dict` with the parameters to replicate this object.

        Returns:
            dict:
                Parameters of this distribution.
        """
        self.check_fit()
        univariates = [univariate.to_dict() for univariate in self.univariates]

        return {
            'covariance': self.covariance.tolist(),
            'univariates': univariates,
            'columns': self.columns,
            'type': get_qualified_name(self),
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the distribution, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Multivariate:
                Instance of the distribution defined on the parameters.
        """
        instance = cls()
        instance.univariates = []
        instance.columns = copula_dict['columns']

        for parameters in copula_dict['univariates']:
            instance.univariates.append(Univariate.from_dict(parameters))

        instance.covariance = np.array(copula_dict['covariance'])
        instance.fitted = True

        return instance
