from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from copulas import check_valid_values, random_state


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


class TestRandomStateDecorator(TestCase):

    @patch('copulas.np.random')
    def test_valid_random_state(self, random_mock):
        """The decorated function use the random_seed attribute if present."""
        # Setup
        my_function = MagicMock()
        instance = MagicMock()
        instance.random_seed = 42

        args = ('some', 'args')
        kwargs = {'keyword': 'value'}

        random_mock.get_state.return_value = "random state"

        # Run
        decorated_function = random_state(my_function)
        decorated_function(instance, *args, **kwargs)

        # Check
        my_function.assert_called_once_with(instance, *args, **kwargs)

        instance.assert_not_called
        random_mock.get_state.assert_called_once_with()
        random_mock.seed.assert_called_once_with(42)
        random_mock.set_state.assert_called_once_with("random state")

    @patch('copulas.np.random')
    def test_no_random_state(self, random_mock):
        """If random_seed is None, the decorated function only call to the original."""
        # Setup
        my_function = MagicMock()
        instance = MagicMock()
        instance.random_seed = None

        args = ('some', 'args')
        kwargs = {'keyword': 'value'}

        random_mock.get_state.return_value = "random state"

        # Run
        decorated_function = random_state(my_function)
        decorated_function(instance, *args, **kwargs)

        # Check
        my_function.assert_called_once_with(instance, *args, **kwargs)

        instance.assert_not_called
        random_mock.get_state.assert_not_called()
        random_mock.seed.assert_not_called()
        random_mock.set_state.assert_not_called()
