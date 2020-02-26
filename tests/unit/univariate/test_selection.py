from unittest import TestCase

import numpy as np
from scipy.stats import truncnorm

from copulas.univariate import GaussianKDE, GaussianUnivariate, TruncatedGaussian
from copulas.univariate.selection import select_univariate


class TestSelectUnivariate(TestCase):

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
        mode1 = np.random.normal(size=size) * mask
        mode2 = np.random.normal(size=size, loc=10) * (1.0 - mask)
        self.bimodal_data = mode1 + mode2

    def test_select_univariate(self):
        """
        Suppose the data follows a bimodal distribution. The model selector should be able to
        figure out that the GaussianKDE is best.
        """
        model = select_univariate(
            self.bimodal_data, [
                GaussianKDE, GaussianUnivariate, TruncatedGaussian])
        assert isinstance(model, GaussianKDE)

    def test_binary(self):
        """
        Suppose the data follows a Bernoulli distribution. The KS statistic should be larger
        for a TruncatedGaussian model than a GaussianKDE model which can somewhat capture a
        Bernoulli distribution as it resembles a bimodal distribution.
        """
        model = select_univariate(self.binary_data, [GaussianKDE(), TruncatedGaussian()])
        assert isinstance(model, GaussianKDE)

    def test_truncated(self):
        """
        Suppose the data follows a truncated normal distribution. The KS statistic should be
        larger for a Gaussian model than a TruncatedGaussian model (since the fit is worse).
        """
        model = select_univariate(self.truncated_data, [GaussianUnivariate(), TruncatedGaussian()])
        assert isinstance(model, TruncatedGaussian)
