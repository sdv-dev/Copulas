from unittest import TestCase

import scipy
import numpy as np
import pandas as pd

from copulas.multivariate.gaussian import GaussianMultivariate
from copulas.univariate import BetaUnivariate, GaussianUnivariate


class TestGaussianCopula(TestCase):

    def test_marginals_gaussian(self):
        size = 10000
        data = pd.DataFrame({
            "x": np.random.normal(size=size),
            "y": np.random.normal(size=size, loc=10.0),
            "z": np.random.normal(size=size, loc=5.0, scale=2.0),
        })
        copula = GaussianMultivariate(distribution=GaussianUnivariate())
        self._test_marginals(data, copula)

    def test_marginals_beta(self):
        size = 10000
        data = pd.DataFrame({
            "w": np.random.beta(1.0, 3.0),
            "x": np.random.exponential(size=size) + 10.0,
            "y": np.random.uniform(size=size) - 5.0,
            "z": np.random.normal(size=size, scale=3),
        })
        copula = GaussianMultivariate(distribution=BetaUnivariate())
        self._test_marginals(data, copula)

    def _test_marginals(self, data, copula):
        """
        This test fits a Gaussian copula to a dataset D, generates some synthetic
        data S, and then uses a statistical test to check whether (1) for each
        column C, the distribution of D[C] and S[C] are similar and (2) for each
        pair of columns C1 and C2 where C1 != C2, the distributions are not similar.
        """
        copula.fit(data)
        synthetic_data = copula.sample(len(data))

        # Test Marginals
        for col_name_1, col_1 in data.iteritems():
            for col_name_2, col_2 in synthetic_data.iteritems():
                if col_name_1 == col_name_2:
                    self._compare_univariate(col_1, col_2)
        
        # Test Pairwise
        for col_1, _ in data.iteritems():
            for col_2, _ in data.iteritems():
                if col_1 == col_2:
                    continue
                X_real = data[[col_1, col_2]].values
                X_synthetic = synthetic_data[[col_1, col_2]].values
                self._compare_bivariate(X_real, X_synthetic)

    def _compare_univariate(self, X_real, X_synthetic):
        """
        This compares the emperical CDFs of the two 1d datasets and computes the KS
        test statistic. If the statistic is too large, then we reject the null hypothesis
        and raise an error suggesting that the distributions might be different.
        """
        statistic, pvalue = scipy.stats.ks_2samp(X_real, X_synthetic)
        if pvalue < 0.05:
            raise ValueError("These distributions might be different.")

    def _compare_bivariate(self, X_real, X_synthetic):
        projection = np.random.random(size=2)
        X_real, X_synthetic = X_real.dot(projection), X_synthetic.dot(projection)
        statistic, pvalue = scipy.stats.ks_2samp(X_real, X_synthetic)
        if pvalue < 0.05:
            raise ValueError("These distributions might be different.")
