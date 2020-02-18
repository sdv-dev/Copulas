from unittest import TestCase

import numpy as np
from scipy.stats import truncnorm

from copulas.univariate import GaussianKDE, GaussianUnivariate, TruncatedGaussian, Univariate
from copulas.univariate.selection import ks_statistic, select_univariate


class TestUnivariateSelection(TestCase):

    def setUp(self):
        size = 1000
        np.random.seed(42)

        # Binary Data
        self.binary_data = np.random.randint(0, 2, size=10000)

        # Truncated Normal
        a, b, loc, scale = -1.0, 0.5, 0.0, 1.0
        self.truncated_data = truncnorm.rvs(a, b, loc=loc, scale=scale, size=10000)

        # Mixture of Normals
        mask = np.random.normal(size=size) > 0.5
        self.bimodal_data = np.random.normal(size=size) * mask + \
            np.random.normal(size=size, loc=10) * (1.0 - mask)

    def test_select_univariate(self):
        """
        Suppose the data follows a bimodal distribution. The model selector should be able to
        figure out that the GaussianKDE is best.
        """
        model = select_univariate(self.bimodal_data)
        assert isinstance(model, GaussianKDE)

    def test_base_univariate(self):
        """
        Suppose the data follows a bimodal distribution. If we use the base Univariate class
        to fit the data, it should automatically select the GaussianKDE.
        """
        model = Univariate()
        model.fit(self.bimodal_data)
        assert isinstance(model._instance, GaussianKDE)

    def test_binary(self):
        """
        Suppose the data follows a Bernoulli distribution. The KS statistic should be larger
        for a TruncatedGaussian model than a GaussianKDE model which can somewhat capture a
        Bernoulli distribution as it resembles a bimodal distribution.
        """
        kde_likelihood = ks_statistic(GaussianKDE(), self.binary_data)
        truncated_likelihood = ks_statistic(TruncatedGaussian(), self.binary_data)
        assert kde_likelihood < truncated_likelihood

    def test_truncated(self):
        """
        Suppose the data follows a truncated normal distribution. The KS statistic should be
        larger for a Gaussian model than a TruncatedGaussian model (since the fit is worse).
        """
        gaussian_likelihood = ks_statistic(GaussianUnivariate(), self.truncated_data)
        truncated_likelihood = ks_statistic(TruncatedGaussian(), self.truncated_data)
        assert truncated_likelihood < gaussian_likelihood

    def test_bimodal(self):
        """
        Suppose the data follows a bimodal distribution. The KS statistic should be larger
        for a Gaussian model than a GaussianKDE model (since it can't capture 2 modes).
        """
        kde_likelihood = ks_statistic(GaussianKDE(), self.bimodal_data)
        gaussian_likelihood = ks_statistic(GaussianUnivariate(), self.bimodal_data)
        assert kde_likelihood < gaussian_likelihood
