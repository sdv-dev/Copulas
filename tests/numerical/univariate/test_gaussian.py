from unittest import TestCase

import numpy as np
from scipy.stats import norm

from copulas.univariate.gaussian import GaussianUnivariate


class TestGaussianUnivariate(TestCase):

    def test_fit_and_sample(self):
        np.random.seed(42)
        self._test_fit_and_sample(loc=10.0, scale=2.0)
        self._test_fit_and_sample(loc=-5.0, scale=3.0)

    def _test_fit_and_sample(self, loc, scale, N=10000):
        """
        Sample N data points from the normal distribution with the given parameters
        and check whether the estimated parameters are close. Then, generate
        some synthetic samples, use them to estimate the parameters, and check
        whether these are still close to the given parameters.
        """
        def _fit_and_sample(data):
            distribution = GaussianUnivariate()
            distribution.fit(data)
            self.assertAlmostEqual(distribution.mean, loc, delta=0.1)
            self.assertAlmostEqual(distribution.std, scale, delta=0.1)
            return distribution.sample(N)

        data = norm.rvs(loc=loc, scale=scale, size=N)
        data = _fit_and_sample(data)
        _fit_and_sample(data)
