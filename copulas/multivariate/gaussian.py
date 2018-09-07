import logging

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import integrate

from copulas.multivariate.base import Multivariate
from copulas.univariate.gaussian import GaussianUnivariate

LOGGER = logging.getLogger(__name__)


class GaussianMultivariate(Multivariate):
    """ Class for a gaussian copula model """

    def __init__(self):
        super().__init__()
        self.distribs = {}
        self.cov_matrix = None
        self.data = None
        self.means = None
        self.pdf = None
        self.cdf = None
        self.ppf = None

    def __str__(self):
        distribs = [
            '\n{}\n==============\n{}'.format(key, value)
            for key, value in self.distribs.items()
        ]

        details = (
            '\n\nCopula Distribution:\n{}'
            '\n\nCovariance matrix:\n{}'
            '\n\nMeans:\n{}'.format(self.distribution, self.cov_matrix, self.means)
        )
        return '\n'.join(distribs) + details

    def fit(self, data, distrib_map=None):
        LOGGER.debug('Fitting Gaussian Copula')
        keys = data.keys()

        # create distributions based on user input
        if distrib_map is not None:
            for key in distrib_map:
                # this isn't fully working yet
                self.distribs[key] = distrib_map[key](data[key])

        else:
            for key in keys:
                self.distribs[key] = GaussianUnivariate()
                self.distribs[key].fit(data[key])

        self.cov_matrix, self.means, self.distribution = self._get_parameters(data)
        self.pdf = st.multivariate_normal.pdf

    def _get_parameters(self, data):
        result = data.copy()

        for column in result.keys():
            X = result[column]
            distrib = self.distribs[column]

            # get original distrib's cdf of the column
            cdf = distrib.cumulative_density(X)

            # get inverse cdf using standard normal
            result[column] = st.norm.ppf(cdf)

        # remove any rows that have infinite values
        result = result[(result != np.inf).all(axis=1)]

        means = list(result.mean(axis=0))
        cov = result.cov()

        return (cov.values, means, result)

    def get_pdf(self, X):
        # make cov positive semi-definite
        covariance = self.cov_matrix * np.identity(self.cov_matrix.shape[0])
        return self.pdf(X, cov=covariance)

    def get_cdf(self, X):
        # Wrapper for pdf to accept vector as args
        def func(*args):
            return self.get_pdf(list(args))

        # Lower bound for integral, to split significant part from tail
        lower_bound = self.get_lower_bound()

        ranges = [[lower_bound, val] for val in X]
        return integrate.nquad(func, ranges)[0]

    def sample(self, num_rows=1):
        res = {}
        means = np.zeros(self.cov_matrix.shape[0])
        s = (num_rows,)

        # clean up cavariance matrix
        clean_cov = np.nan_to_num(self.cov_matrix)
        samples = np.random.multivariate_normal(means, clean_cov, size=s)
        # run through cdf and inverse cdf
        for i, (label, distrib) in enumerate(self.distribs.items()):
            # use standard normal's cdf
            res[label] = st.norm.cdf(samples[:, i])

            # use original distributions inverse cdf
            res[label] = distrib.percent_point(res[label])
        return pd.DataFrame(data=res)

    def to_dict(self):
        distributions = {
            name: distribution.to_dict() for name, distribution in self.distribs.items()
        }

        return {
            'means': self.means,
            'cov_matrix': self.cov_matrix.tolist(),
            'distribs': distributions
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Set attributes with provided values."""
        instance = cls()
        instance.distribs = {}

        for name, parameters in copula_dict['distribs'].items():
            instance.distribs[name] = GaussianUnivariate.from_dict(parameters)

        instance.cov_matrix = np.array(copula_dict['cov_matrix'])
        instance.means = copula_dict['means']
        return instance

    def get_lower_bound(self):
        lower_bounds = []

        for distribution in self.distribs.values():
            lower_bound = distribution.inverse_cdf(distribution.mean / 10000)
            if not pd.isnull(lower_bound):
                lower_bounds.append(lower_bound)

        return min(lower_bounds)
