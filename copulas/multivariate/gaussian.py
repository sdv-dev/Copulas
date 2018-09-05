import logging

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.stats as st

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

    def _get_parameters(self):
        result = self.data.copy()

        for column in result.keys():
            X = result[column]
            distrib = self.distribs[column]

            # get original distrib's cdf of the column
            cdf = distrib.get_cdf(X)

            # get inverse cdf using standard normal
            result[column] = st.norm.ppf(cdf)

        # remove any rows that have infinite values
        result = result[(result != np.inf).all(axis=1)]

        means = list(result.mean(axis=0))
        cov = result.cov()

        return (cov.values, means, result)

    def fit(self, data, distrib_map=None):
        LOGGER.debug('Fitting Gaussian Copula')
        self.data = data
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

        self.cov_matrix, self.means, self.distribution = self._get_parameters()
        self.pdf = st.multivariate_normal.pdf

    def get_pdf(self, X):
        # make cov positive semi-definite
        cov = self.cov_matrix * np.identity(3)
        return self.pdf(X, self.means, cov)

    def get_cdf(self, X):
        def func(*args):
            return self.get_pdf([args[i] for i in range(len(args))])

        # TODO: fix lower bounds
        ranges = [[-10000, val] for val in X]

        return integrate.nquad(func, ranges)[0]

    def sample(self, num_rows=1):
        res = {}
        means = np.zeros(len(self.means))
        s = (num_rows,)

        # clean up cavariance matrix
        clean_cov = np.nan_to_num(self.cov_matrix)
        samples = np.random.multivariate_normal(means, clean_cov, size=s)
        # run through cdf and inverse cdf
        for i in range(self.data.shape[1]):
            label = self.data.iloc[:, i].name
            distrib = self.distribs[label]

            # use standard normal's cdf
            res[label] = st.norm.cdf(samples[:, i])

            # use original distributions inverse cdf
            res[label] = distrib.inverse_cdf(res[label])

        return pd.DataFrame(data=res)
