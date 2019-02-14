from unittest import TestCase

import numpy as np

from copulas.univariate.base import Univariate
from tests import compare_nested_iterables


class TestUnivariate(TestCase):

    def test_get_constant_value(self):
        """get_constant_value return the unique value of an array if it exists."""
        # Setup
        X = np.array([1, 1, 1, 1])
        expected_result = 1

        # Run
        result = Univariate._get_constant_value(X)

        # Check
        assert result == expected_result

    def test_get_constant_value_non_constant(self):
        """get_constant_value return None on non-constant arrays."""
        # Setup
        X = np.array(range(5))
        expected_result = None

        # Run
        result = Univariate._get_constant_value(X)

        # Check
        assert result is expected_result

    def test__sample_sample(self):
        """_constant_sample returns a constant array of num_samples length."""
        # Setup
        instance = Univariate()
        instance.constant_value = 15

        expected_result = np.array([15, 15, 15, 15, 15])

        # Run
        result = instance._constant_sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

    def test__constant_cumulative_distribution(self):
        """constant_cumulative_distribution returns only 0 and 1."""
        # Setup
        instance = Univariate()
        instance.constant_value = 3

        X = np.array([1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 1, 1, 1])

        # Run
        result = instance._constant_cumulative_distribution(X)

        # Check
        compare_nested_iterables(result, expected_result)

    def test__constant_probability_density(self):
        """constant_probability_density only is 1 in self.constant_value."""
        # Setup
        instance = Univariate()
        instance.constant_value = 3

        X = np.array([1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 1, 0, 0])

        # Run
        result = instance._constant_probability_density(X)

        # Check
        compare_nested_iterables(result, expected_result)

    def test__constant_percent_point(self):
        """constant_percent_point only is self.constant_value in non-zero probabilities."""
        # Setup
        instance = Univariate()
        instance.constant_value = 3

        X = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_result = np.array([3, 3, 3, 3, 3])

        # Run
        result = instance._constant_percent_point(X)

        # Check
        compare_nested_iterables(result, expected_result)
