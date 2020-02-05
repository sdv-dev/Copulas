from unittest import TestCase

import numpy as np
from scipy.stats import beta

from copulas.univariate import BetaUnivariate


class TestBetaUnivariate(TestCase):

    def test___init__(self):
        """On init, default values are set on instance."""
        copula = BetaUnivariate()
        assert copula.a is None
        assert copula.b is None
        assert copula.loc is None
        assert copula.scale is None

    def test_fit(self):
        """On fit, stats from fit data are set in the model."""

        # Generate data with known parameters
        a, b, loc, scale = 1.0, 1.0, 10.0, 11.0
        data = beta.rvs(a, b, loc, scale, size=100000)

        # Fit the model and check parameters
        copula = BetaUnivariate()
        copula.fit(data)
        self.assertAlmostEqual(copula.a, a, places=1)
        self.assertAlmostEqual(copula.b, b, places=1)
        self.assertAlmostEqual(copula.loc, loc, places=1)
        self.assertAlmostEqual(copula.scale, scale, places=1)

    def test_test_fit_equal_values(self):
        """If it's fit with constant data, contant_value is set."""
        instance = BetaUnivariate()
        instance.fit(np.array([5, 5, 5, 5, 5, 5]))
        assert instance.a is None
        assert instance.b is None
        assert instance.constant_value == 5

    def test_valid_serialization_unfit_model(self):
        """For a unfitted model to_dict and from_dict are opposites."""
        instance = BetaUnivariate()
        result = BetaUnivariate.from_dict(instance.to_dict())
        assert instance.to_dict() == result.to_dict()

    def test_valid_serialization_fit_model(self):
        """For a fitted model to_dict and from_dict are opposites."""
        instance = BetaUnivariate()
        instance.fit(np.array([1, 2, 3, 2, 1]))
        result = BetaUnivariate.from_dict(instance.to_dict())
        assert instance.to_dict() == result.to_dict()
