from unittest import TestCase

import numpy as np
from scipy.stats import uniform

from copulas.univariate.uniform import UniformUnivariate


class TestUniformUnivariate(TestCase):

    def test__fit_constant(self):
        distribution = UniformUnivariate()
        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'loc': 1,
            'scale': 0
        }

    def test__fit(self):
        distribution = UniformUnivariate()

        data = uniform.rvs(size=1000, loc=0, scale=1)
        distribution._fit(data)

        expected = {
            'loc': 0,
            'scale': 1,
        }

        for key, value in distribution._params.items():
            np.testing.assert_allclose(value, expected[key], atol=0.3)

    def test__is_constant_true(self):
        distribution = UniformUnivariate()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = UniformUnivariate()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()
