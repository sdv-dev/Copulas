from unittest import TestCase

import numpy as np
from scipy.stats import gamma

from copulas.univariate import GammaUnivariate


class TestGammaUnivariate(TestCase):

    def test__fit_constant(self):
        distribution = GammaUnivariate()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'a': 0,
            'loc': 1,
            'scale': 0
        }

    def test__fit(self):
        distribution = GammaUnivariate()

        data = gamma.rvs(size=10000, a=1, loc=1, scale=1)
        distribution._fit(data)

        expected = {
            'loc': 1,
            'scale': 1,
            'a': 1,
        }
        for key, value in distribution._params.items():
            np.testing.assert_allclose(value, expected[key], atol=0.3)

    def test__is_constant_true(self):
        distribution = GammaUnivariate()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = GammaUnivariate()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()
