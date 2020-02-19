from unittest import TestCase

import numpy as np
from scipy.stats import truncnorm

from copulas.univariate.truncated_gaussian import TruncatedGaussian


class TestTruncatedGaussian(TestCase):

    def test_fit_and_sample(self):
        """
        The standard form of this distribution is a standard normal truncated to the range [a, b]
        but shifted by loc and divided by scale. Therefore, `a` and `b` are in the domain of a
        standard normal (i.e. reasonable values are in the range [-3, 3]) whereas `loc` and `scale`
        can be any real number.
        """
        np.random.seed(42)
        self._test_fit_and_sample(a=-3.0, b=1.0, loc=0.0, scale=1.0)
        self._test_fit_and_sample(a=-1.0, b=1.0, loc=10.0, scale=3.0)
        self._test_fit_and_sample(a=-2.0, b=3.0, loc=-5.0, scale=1.0)
        self._test_fit_and_sample(a=-2.0, b=3.0, loc=-2.3, scale=4.0)

    def _test_fit_and_sample(self, a, b, loc, scale, N=10000):
        """
        Sample N data points from the truncated normal distribution with the
        given parameters and check whether the estimated parameters are close.
        Then, generate some synthetic samples, use them to estimate the parameters,
        and check whether these are still close to the given parameters.
        """
        def _fit_and_sample(data):
            distribution = TruncatedGaussian()
            distribution.fit(data)
            self.assertAlmostEqual(distribution.a, a, delta=0.5)
            self.assertAlmostEqual(distribution.b, b, delta=0.5)
            self.assertAlmostEqual(distribution.mean, loc, delta=0.5)
            self.assertAlmostEqual(distribution.std, scale, delta=0.5)
            return distribution.sample(N)

        data = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=N)
        data = _fit_and_sample(data)
        _fit_and_sample(data)
