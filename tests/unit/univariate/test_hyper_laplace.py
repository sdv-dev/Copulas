from unittest import TestCase

import numpy as np
from scipy.stats import norm

from copulas.univariate.hyper_laplace import HyperLaplace


class TestHyperLaplace(TestCase):

    def test__fit_constant(self):
        distribution = HyperLaplace()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'loc': 1,
            'scale': 0,
            'a': 1
        }

    def test__fit(self): 
        distribution = HyperLaplace()

        data = norm.rvs(size=100000, loc=0, scale=1)
        distribution._fit(data)

        assert distribution._params['loc'] == 0
        assert distribution._params['scale'] >= 1.90 and distribution._params['scale'] <= 2.10
        assert distribution._params['a'] >= 0.48 and distribution._params['a'] <= 0.52

    def test__is_constant_true(self):
        distribution = HyperLaplace()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = HyperLaplace()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()
