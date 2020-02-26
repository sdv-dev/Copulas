from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from copulas.univariate.gaussian import GaussianUnivariate
from tests import compare_nested_iterables


class TestGaussianUnivariate(TestCase):

    def test___init__(self):
        """On init, default values are set on instance."""

        # Setup / Run
        copula = GaussianUnivariate()

        # Check
        assert not copula.name
        assert copula.mean is None
        assert copula.std is None
        assert copula.constant_value is None

    def test___str__(self):
        """str returns details about the model."""

        # Setup
        copula = GaussianUnivariate()
        expected_result = '\n'.join([
            'Distribution Type: Gaussian',
            'Variable name: None',
            'Mean: None',
            'Standard deviation: None'
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

        # Run
        copula.fit(column)

        # Check
        assert copula.mean == 2.5
        assert copula.std == 1.707825127659933
        assert copula.name == 'column'
        assert copula.fitted is True

    def test_test_fit_equal_values(self):
        """If it's fit with constant data, contant_value is set."""

        # Setup
        instance = GaussianUnivariate()
        column = np.array([5, 5, 5, 5, 5, 5])

        # Run
        instance.fit(column)

        # Check
        assert instance.mean is None
        assert instance.std is None
        assert instance.constant_value == 5

    def test_get_probability_density(self):
        """Probability_density returns the normal probability distribution for the given values."""

        # Setup
        copula = GaussianUnivariate()
        column = np.array([-1, 0, 1])
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
        column = np.array([-1, 0, 1])
        copula.fit(column)
        x = pd.Series([0])
        expected_result = [0.5]

        # Run
        result = copula.cumulative_distribution(x)

        # Check
        assert (result == expected_result).all()

    def test_cumulative_distribution_constant(self):
        """cumulative_distribution can be computed for constant distribution."""
        # Setup
        instance = GaussianUnivariate()
        instance.constant_value = 3
        instance._replace_constant_methods()
        instance.fitted = True

        X = np.array([1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 1, 1, 1])

        # Run
        result = instance.cumulative_distribution(X)

        # Check
        compare_nested_iterables(result, expected_result)

    def test_percent_point(self):
        """Percent_point returns the original point from the cumulative probability value."""

        # Setup
        copula = GaussianUnivariate()
        column = np.array([-1, 0, 1])
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
        column = np.array([-1, 0, 1])
        copula.fit(column)
        initial_value = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])

        # Run
        result_a = copula.percent_point(copula.cumulative_distribution(initial_value))
        result_b = copula.cumulative_distribution(copula.percent_point(initial_value))

        # Check
        assert (initial_value - result_a < 10E-5).all()
        assert (initial_value - result_b < 10E-5).all()

    @patch('copulas.univariate.gaussian.np.random.normal')
    def test_sample(self, random_mock):
        """After fitting, GaussianUnivariate is able to sample new data."""
        # Setup
        instance = GaussianUnivariate()
        column = np.array([-1, 0, 1])
        instance.fit(column)

        expected_result = np.array([1, 2, 3, 4, 5])
        random_mock.return_value = expected_result

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

        assert instance.mean == 0.0
        assert instance.std == 0.816496580927726
        random_mock.assert_called_once_with(0.0, 0.816496580927726, 5)

    def test_sample_random_state(self):
        """When random state is set, samples are the same."""
        # Setup
        instance = GaussianUnivariate(random_seed=0)
        X = np.array([1, 2, 3, 4, 5])
        instance.fit(X)

        expected_result = np.array([5.494746752403546, 3.565907751154284, 4.384144531132039])

        # Run
        result = instance.sample(3)

        # Check
        assert (result == expected_result).all()

    def test_sample_constant(self):
        """samples can be generated for constant distribution."""
        # Setup
        instance = GaussianUnivariate()
        instance.constant_value = 3
        instance._replace_constant_methods()
        instance.fitted = True

        expected_result = np.array([3, 3, 3, 3, 3])

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

    def test_to_dict(self):
        """To_dict returns the defining parameters of a distribution in a dict."""
        # Setup
        copula = GaussianUnivariate()
        column = np.array([0, 1, 2, 3, 4, 5])
        copula.fit(column)
        expected_result = {
            'type': 'copulas.univariate.gaussian.GaussianUnivariate',
            'mean': 2.5,
            'std': 1.707825127659933,
            'fitted': True,
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
