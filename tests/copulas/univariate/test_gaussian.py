from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.univariate.gaussian import GaussianUnivariate


class TestGaussianUnivariate(TestCase):

    def test___init__(self):
        """On init, default values are set on instance."""

        # Setup / Run
        copula = GaussianUnivariate()

        # Check
        assert not copula.column
        assert copula.mean == 0
        assert copula.std == 1
        assert copula.min == -np.inf
        assert copula.max == np.inf

    def test___str__(self):
        """str returns details about the model."""

        # Setup
        copula = GaussianUnivariate()
        expected_result = '\n'.join([
            'Distribution Type: Gaussian',
            'Variable name: None',
            'Mean: 0',
            'Standard deviation: 1',
            'Max: inf',
            'Min: -inf'
        ])

        # Run
        result = copula.__str__()

        # Check
        assert result == expected_result

    def test_fit(self):
        """On fit, stats from fit data are set in the model."""

        # Setup
        copula = GaussianUnivariate()
        column = [0, 1, 2, 3, 4, 5]
        mean = 2.5
        std = 1.707825127659933
        max_value = 5
        min_value = 0

        # Run
        copula.fit(column)

        # Check
        assert copula.column == column
        assert copula.mean == mean
        assert copula.std == std
        assert copula.min == min_value
        assert copula.max == max_value

    def test_fit_empty_data(self):
        """On fit, if column is empty an error is raised."""

        # Setup
        copula = GaussianUnivariate()
        column = []

        # Run
        with self.assertRaises(ValueError):
            copula.fit(column)

    def test_test_fit_equal_values(self):
        """On fit, even if column has equal values, std is never 0."""

        # Setup
        copula = GaussianUnivariate()
        column = [1, 1, 1, 1, 1, 1]

        # Run
        copula.fit(column)

        # Check
        assert copula.mean == 1
        assert copula.std == 0.001
        assert copula.max == 1
        assert copula.min == 1

    def test_get_pdf(self):
        """get_pdf returns the normal probability distribution value for the given values."""

        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)
        expected_result = 0.48860251190292

        # Run
        result = copula.get_pdf(0)

        # Check
        assert result == expected_result

    def test_get_cdf(self):
        """get_cdf returns the cumulative distribution function value for a point."""

        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)
        x = pd.Series([0])
        expected_result = [0.5]

        # Run
        result = copula.get_cdf(x)

        # Check
        assert (result == expected_result).all()

    def test_inverse_cdf(self):
        """inverse_cdf returns the original point from the cumulative probability value """

        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)
        x = 0.5
        expected_result = 0

        # Run
        result = copula.inverse_cdf(x)

        # Check
        assert (result == expected_result).all()

    def test_inverse_cdf_reverse_get_cdf(self):
        """Combined get_cdf and inverse_cdf is the identity function."""

        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)
        initial_value = pd.Series([0])

        # Run
        result_a = copula.inverse_cdf(copula.get_cdf(initial_value))
        result_b = copula.get_cdf(copula.inverse_cdf(initial_value))

        # Check
        assert (initial_value == result_a).all()
        assert (initial_value == result_b).all()

    def test_sample(self):
        """After fitting, GaussianUnivariate is able to sample new data."""
        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)

        # Run
        result = copula.sample(1000000)

        # Check
        assert len(result) == 1000000
        assert abs(np.mean(result) - copula.mean) < 10E-3
        assert abs(np.std(result) - copula.std) < 10E-3

    def test_to_dict(self):
        """To_dict returns the defining parameters of a distribution in a dict."""
        # Setup
        copula = GaussianUnivariate()
        column = [0, 1, 2, 3, 4, 5]
        copula.fit(column)
        expected_result = {
            'mean': 2.5,
            'std': 1.707825127659933
        }

        # Run
        result = copula.to_dict()

        # Check
        assert result == expected_result

    def test_from_dict(self):
        """From_dict sets the values of a dictionary as attributes of the instance."""
        # Setup
        parameters = {
            'mean': 2.5,
            'std': 1.707825127659933
        }

        # Run
        copula = GaussianUnivariate.from_dict(parameters)

        # Check
        assert copula.mean == 2.5
        assert copula.std == 1.707825127659933

        copula.sample(10)
