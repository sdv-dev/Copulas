from unittest import TestCase

import numpy as np

from copulas.optimize import bisect, chandrupatla


class TestOptimize(TestCase):

    def test_uniform(self):
        """Find the zero of a line."""
        N = 100
        target = np.random.random(size=N)

        def _f(x):
            return x - target
        for optimizer in [bisect, chandrupatla]:
            with self.subTest(optimizer=optimizer):
                x = optimizer(_f, np.zeros(shape=N), np.ones(shape=N))
                assert np.abs(x - target).max() < 1e-6

    def test_polynomial(self):
        """Find the zero of a polynomial."""
        def _f(x):
            return np.power(x - 10.0, 3.0)
        for optimizer in [bisect, chandrupatla]:
            with self.subTest(optimizer=optimizer):
                x = optimizer(_f, np.array([0.0]), np.array([100.0]))
                assert np.abs(x - 10.0).max() < 1e-6
