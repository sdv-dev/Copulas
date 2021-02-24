from unittest import TestCase

import numpy as np
from scipy.stats import loglaplace

from copulas.univariate import LogLaplace


class TestLogLaplaceUnivariate(TestCase):

    def test__fit_constant(self):
        distribution = LogLaplace()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'c': 2,
            'loc': 1,
            'scale': 0
        }

    def test__fit(self):
        distribution = LogLaplace()

        data = loglaplace.rvs(size=10000, c=2, loc=1, scale=1)
        distribution._fit(data)

        expected = {
            'loc': 1,
            'scale': 1,
            'c': 2,
        }
        for key, value in distribution._params.items():
            np.testing.assert_allclose(value, expected[key], atol=0.3)

    def test__is_constant_true(self):
        distribution = LogLaplace()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = LogLaplace()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()

    def test__extract_constant(self):
        distribution = LogLaplace()
        distribution._params = {
            'c': 2,
            'loc': 1,
            'scale': 0
        }

        constant = distribution._extract_constant()

        assert 1 == constant
