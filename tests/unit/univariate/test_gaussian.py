from unittest import TestCase

import numpy as np
from scipy.stats import norm

from copulas.univariate.gaussian import GaussianUnivariate


class TestGaussianUnivariate(TestCase):

    def test__fit_constant(self):
        distribution = GaussianUnivariate()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'loc': 1,
            'scale': 0
        }

    def test__fit(self):
        distribution = GaussianUnivariate()

        data = norm.rvs(size=1000, loc=1, scale=1)
        distribution._fit(data)

        assert distribution._params == {
            'loc': np.mean(data),
            'scale': np.std(data),
        }

    def test__is_constant_true(self):
        distribution = GaussianUnivariate()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = GaussianUnivariate()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()
