from unittest import TestCase

import numpy as np
from scipy.stats import gamma

from copulas.univariate import GammaUnivariate


class TestGammaUnivariate(TestCase):

    def test_fit_and_sample(self):
        np.random.seed(42)
        self._test_fit_and_sample(a=1.0, loc=10.0, scale=2.0)
        self._test_fit_and_sample(a=3.0, loc=-1.0, scale=0.7)

    def _test_fit_and_sample(self, a, loc, scale, N=10000):
        """
        Sample N data points from the gamma distribution with the given parameters
        and check whether the estimated parameters are close. Then, generate
        some synthetic samples, use them to estimate the parameters, and check
        whether these are still close to the given parameters.
        """
        def _fit_and_sample(data):
            distribution = GammaUnivariate()
            distribution.fit(data)
            self.assertAlmostEqual(distribution.a, a, delta=0.5)
            self.assertAlmostEqual(distribution.loc, loc, delta=0.5)
            self.assertAlmostEqual(distribution.scale, scale, delta=0.5)
            return distribution.sample(N)

        data = gamma.rvs(a, loc, scale, size=N)
        data = _fit_and_sample(data)
        _fit_and_sample(data)
