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
        assert not copula.name
        assert copula.mean == 0
        assert copula.std == 1

    def test___str__(self):
        """str returns details about the model."""

        # Setup
        copula = GaussianUnivariate()
        expected_result = '\n'.join([
            'Distribution Type: Gaussian',
            'Variable name: None',
            'Mean: 0',
            'Standard deviation: 1'
        ])

        # Run
        result = copula.__str__()

        # Check
        assert result == expected_result

    def test_fit(self):
        """On fit, stats from fit data are set in the model."""

        # Setup
        copula = GaussianUnivariate()
        column = pd.Series([0, 1, 2, 3, 4, 5], name='column')
        mean = 2.5
        std = 1.707825127659933
        name = 'column'

        # Run
        copula.fit(column)

        # Check
        assert copula.mean == mean
        assert copula.std == std
        assert copula.name == name
        assert copula.fitted

    def test_fit_empty_data(self):
        """On fit, if column is empty an error is raised."""

        # Setup
        copula = GaussianUnivariate()
        column = pd.Series([])

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

    def test_get_probability_density(self):
        """Probability_density returns the normal probability distribution for the given values."""

        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)
        expected_result = 0.48860251190292

        # Run
        result = copula.probability_density(0)

        # Check
        assert result == expected_result

    def test_cumulative_distribution(self):
        """Cumulative_density returns the cumulative distribution value for a point."""

        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)
        x = pd.Series([0])
        expected_result = [0.5]

        # Run
        result = copula.cumulative_distribution(x)

        # Check
        assert (result == expected_result).all()

    def test_percent_point(self):
        """Percent_point returns the original point from the cumulative probability value."""

        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)
        x = 0.5
        expected_result = 0

        # Run
        result = copula.percent_point(x)

        # Check
        assert (result == expected_result).all()

    def test_percent_point_reverse_cumulative_distribution(self):
        """Combined cumulative_distribution and percent_point is the identity function."""

        # Setup
        copula = GaussianUnivariate()
        column = [-1, 0, 1]
        copula.fit(column)
        initial_value = pd.Series([0])

        # Run
        result_a = copula.percent_point(copula.cumulative_distribution(initial_value))
        result_b = copula.cumulative_distribution(copula.percent_point(initial_value))

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
            'type': 'copulas.univariate.gaussian.GaussianUnivariate',
            'mean': 2.5,
            'std': 1.707825127659933,
            'fitted': True
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
            'std': 1.707825127659933,
            'fitted': True,
        }

        # Run
        copula = GaussianUnivariate.from_dict(parameters)

        # Check
        assert copula.mean == 2.5
        assert copula.std == 1.707825127659933
        assert copula.fitted

        copula.sample(10)

    def test_valid_serialization_unfit_model(self):
        """For a unfitted model to_dict and from_dict are opposites."""
        # Setup
        instance = GaussianUnivariate()

        # Run
        result = GaussianUnivariate.from_dict(instance.to_dict())

        # Check
        assert instance.to_dict() == result.to_dict()

    def test_valid_serialization_fit_model(self):
        """For a fitted model to_dict and from_dict are opposites."""
        # Setup
        instance = GaussianUnivariate()
        instance.fitted = True
        instance.std = 1
        instance.mean = 5

        # Run
        result = GaussianUnivariate.from_dict(instance.to_dict())

        # Check
        assert instance.to_dict() == result.to_dict()
