from unittest import TestCase

import numpy as np
from scipy.stats import gamma

from copulas.univariate import GammaUnivariate


class TestGammaUnivariate(TestCase):

    def test___init__(self):
        """On init, default values are set on instance."""
        copula = GammaUnivariate()
        assert copula.a is None
        assert copula.loc is None
        assert copula.scale is None

    def test_fit(self):
        """On fit, stats from fit data are set in the model."""

        # Generate data with known parameters
        a, loc, scale = 1.0, 3.0, 5.0
        data = gamma.rvs(a, loc, scale, size=100000)

        # Fit the model and check parameters
        copula = GammaUnivariate()
        copula.fit(data)
        self.assertAlmostEqual(copula.a, a, places=1)
        self.assertAlmostEqual(copula.loc, loc, places=1)
        self.assertAlmostEqual(copula.scale, scale, places=1)

    def test_test_fit_equal_values(self):
        """If it's fit with constant data, contant_value is set."""
        instance = GammaUnivariate()
        instance.fit(np.array([5, 5, 5, 5, 5, 5]))
        assert instance.constant_value == 5

    def test_valid_serialization_unfit_model(self):
        """For a unfitted model to_dict and from_dict are opposites."""
        instance = GammaUnivariate()
        result = GammaUnivariate.from_dict(instance.to_dict())
        assert instance.to_dict() == result.to_dict()

    def test_valid_serialization_fit_model(self):
        """For a fitted model to_dict and from_dict are opposites."""
        instance = GammaUnivariate()
        instance.fit(np.array([1, 2, 3, 2, 1]))
        result = GammaUnivariate.from_dict(instance.to_dict())
        assert instance.to_dict() == result.to_dict()
