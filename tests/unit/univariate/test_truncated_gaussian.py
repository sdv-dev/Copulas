from unittest import TestCase

import numpy as np
from scipy.stats import truncnorm

from copulas.univariate.truncated_gaussian import TruncatedGaussian


class TestTruncatedGaussian(TestCase):

    def test__fit_constant(self):
        distribution = TruncatedGaussian()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'a': 1,
            'b': 1,
            'loc': 1,
            'scale': 0
        }

    def test__fit(self):
        distribution = TruncatedGaussian()

        data = truncnorm.rvs(size=10000, a=0, b=3, loc=3, scale=1)
        distribution._fit(data)

        expected = {
            'loc': 3,
            'scale': 1,
            'a': 0,
            'b': 3
        }
        for key, value in distribution._params.items():
            np.testing.assert_allclose(value, expected[key], atol=0.3)

    def test__is_constant_true(self):
        distribution = TruncatedGaussian()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = TruncatedGaussian()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()
