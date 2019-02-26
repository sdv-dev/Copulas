from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_array_equal

from copulas import random_state, scalarize, vectorize


class TestVectorize(TestCase):

    def test_1d_array(self):
        """When applied to a function it allows it to work with 1-d vectors."""
        # Setup
        function = MagicMock()
        function.return_value = 1
        function.__doc__ = 'Docstring of the original function.'

        instance = MagicMock()

        vector = np.array([1, 2, 3])
        args = ['positional', 'arguments']
        kwargs = {
            'keyword': 'arguments'
        }

        expected_result = np.ones(3)
        expected_function_call_args_list = [
            ((instance, 1, 'positional', 'arguments'), {'keyword': 'arguments'}),
            ((instance, 2, 'positional', 'arguments'), {'keyword': 'arguments'}),
            ((instance, 3, 'positional', 'arguments'), {'keyword': 'arguments'})
        ]

        # Run (Decorator)
        vectorized_function = vectorize(function)

        # Check (Decorator)
        assert callable(vectorized_function)
        assert vectorized_function.__doc__ == 'Docstring of the original function.'

        # Run (Decorated function)
        result = vectorized_function(instance, vector, *args, **kwargs)

        # Check (Result of decorated function call)
        assert result.shape == (3,)
        assert_array_equal(result, expected_result)

        assert function.call_args_list == expected_function_call_args_list

        instance.assert_not_called()
        assert instance.method_calls == []

    def test_2d_array(self):
        """When applied to a function it allows it to work with 2-d vectors."""
        # Setup
        function = MagicMock()
        function.return_value = 1
        function.__doc__ = 'Docstring of the original function.'

        instance = MagicMock()

        vector = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        args = ['positional', 'arguments']
        kwargs = {
            'keyword': 'arguments'
        }

        expected_result = np.ones(3)
        expected_function_call_args_list = [
            ((instance, 1, 2, 3, 'positional', 'arguments'), {'keyword': 'arguments'}),
            ((instance, 4, 5, 6, 'positional', 'arguments'), {'keyword': 'arguments'}),
            ((instance, 7, 8, 9, 'positional', 'arguments'), {'keyword': 'arguments'})
        ]

        # Run (Decorator)
        vectorized_function = vectorize(function)

        # Check (Decorator)
        assert callable(vectorized_function)
        assert vectorized_function.__doc__ == 'Docstring of the original function.'

        # Run (Decorated function)
        result = vectorized_function(instance, vector, *args, **kwargs)

        # Check (Result of decorated function call)
        assert result.shape == (3,)
        assert_array_equal(result, expected_result)

        assert function.call_args_list == expected_function_call_args_list

        instance.assert_not_called()
        assert instance.method_calls == []

    def test_raises_valueerror(self):
        """If given an array of dimensionality higher than 2 a ValueError is raised."""
        # Setup
        function = MagicMock()
        X = np.array([
            [[1, 2, 3]]
        ])
        instance = MagicMock()
        args = ()
        kwargs = {}

        # Run
        vectorized_function = vectorize(function)

        # Check
        with self.assertRaises(ValueError):
            vectorized_function(instance, X, *args, **kwargs)


class TestScalarize(TestCase):

    def test_decorator(self):
        """When applied to a function it allows it to work with scalars."""
        # Setup
        function = MagicMock()
        function.__doc__ = 'Docstring of the original function.'
        function.return_value = np.array(['return_value'])

        instance = MagicMock()
        args = ['positional', 'arguments']
        kwargs = {
            'keyword': 'arguments'
        }

        expected_result = 'return_value'

        # Run (Decorator)
        scalarized_function = scalarize(function)

        # Check (Decorator)
        assert callable(scalarized_function)
        assert scalarized_function.__doc__ == 'Docstring of the original function.'

        # Run (Decorated function)
        result = scalarized_function(instance, 0, *args, **kwargs)

        # Check (Decorated function)
        assert result == expected_result

        function.assert_called_once_with(instance, np.array([0]), *args, **kwargs)

        instance.assert_not_called()
        assert instance.method_calls == []


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
