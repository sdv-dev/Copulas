from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from copulas import check_valid_values


class TestCheckValidValues(TestCase):

    def test_check_valid_values_raises_valuerror_if_nans(self):
        """check_valid_values raises a ValueError if is given data with nans."""
        # Setup
        X = np.array([
            [1.0, np.nan],
            [0.0, 1.0]
        ])

        instance_mock = MagicMock()
        function_mock = MagicMock()

        # Run
        decorated_function = check_valid_values(function_mock)

        # Check:
        with self.assertRaises(ValueError):
            decorated_function(instance_mock, X)

        function_mock.assert_not_called()
        instance_mock.assert_not_called()

    def test_check_valid_values_raises_valueerror_if_not_numeric(self):
        """check_valid_values raises a ValueError if is given data with non numeric values."""
        # Setup
        X = np.array([
            [1.0, 'A'],
            [0.0, 1.0]
        ])

        instance_mock = MagicMock()
        function_mock = MagicMock()

        # Run
        decorated_function = check_valid_values(function_mock)

        # Check:
        with self.assertRaises(ValueError):
            decorated_function(instance_mock, X)

        function_mock.assert_not_called()
        instance_mock.assert_not_called()

    def test_check_valid_values_raises_valueerror_empty_dataset(self):
        """check_valid_values raises a ValueError if given data is empty."""
        # Setup
        X = np.array([])

        instance_mock = MagicMock()
        function_mock = MagicMock()

        # Run
        decorated_function = check_valid_values(function_mock)

        # Check:
        with self.assertRaises(ValueError):
            decorated_function(instance_mock, X)

        function_mock.assert_not_called()
        instance_mock.assert_not_called()
