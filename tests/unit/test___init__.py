import sys
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import copulas
from copulas import (
    _find_addons, check_valid_values, get_instance, random_state, scalarize, validate_random_state,
    vectorize)
from copulas.multivariate import GaussianMultivariate


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

        # Run Decorator
        vectorized_function = vectorize(function)

        # Check Decorator
        assert callable(vectorized_function)
        assert vectorized_function.__doc__ == 'Docstring of the original function.'

        # Run decorated function
        result = vectorized_function(instance, vector, *args, **kwargs)

        # Check result of decorated function call
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

        # Run Decorator
        vectorized_function = vectorize(function)

        # Check Decorator
        assert callable(vectorized_function)
        assert vectorized_function.__doc__ == 'Docstring of the original function.'

        # Run decorated function
        result = vectorized_function(instance, vector, *args, **kwargs)

        # Check result of decorated function call
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
        error_msg = 'Arrays of dimensionality higher than 2 are not supported.'
        with pytest.raises(ValueError, match=error_msg):
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

        # Run Decorator
        scalarized_function = scalarize(function)

        # Check Decorator
        assert callable(scalarized_function)
        assert scalarized_function.__doc__ == 'Docstring of the original function.'

        # Run decorated function
        result = scalarized_function(instance, 0, *args, **kwargs)

        # Check decorated function
        assert result == expected_result

        function.assert_called_once_with(instance, np.array([0]), *args, **kwargs)

        instance.assert_not_called()
        assert instance.method_calls == []


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
        error_msg = 'There are nan values in your data.'
        with pytest.raises(ValueError, match=error_msg):
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
        error_msg = 'There are non-numerical values in your data.'
        with pytest.raises(ValueError, match=error_msg):
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
        error_msg = 'Your dataset is empty.'
        with pytest.raises(ValueError, match=error_msg):
            decorated_function(instance_mock, X)

        function_mock.assert_not_called()
        instance_mock.assert_not_called()


class TestRandomStateDecorator(TestCase):

    @patch('copulas.np.random')
    def test_valid_random_state(self, random_mock):
        """The decorated function use the random_state attribute if present."""
        # Setup
        my_function = MagicMock()
        instance = MagicMock()
        random_state_mock = MagicMock()
        random_state_mock.get_state.return_value = 'desired random state'
        instance.random_state = random_state_mock

        args = ('some', 'args')
        kwargs = {'keyword': 'value'}

        random_mock.get_state.return_value = 'random state'

        # Run
        decorated_function = random_state(my_function)
        decorated_function(instance, *args, **kwargs)

        # Check
        my_function.assert_called_once_with(instance, *args, **kwargs)

        instance.assert_not_called
        random_mock.get_state.assert_has_calls([call(), call()])
        random_mock.get_state.call_count == 2
        random_mock.RandomState.assert_has_calls(
            [call(), call().set_state('random state')])
        random_mock.set_state.assert_has_calls(
            [call('desired random state'), call('random state')])
        assert random_mock.set_state.call_count == 2

    @patch('copulas.np.random')
    def test_no_random_state(self, random_mock):
        """If random_state is None, the decorated function only call to the original."""
        # Setup
        my_function = MagicMock()
        instance = MagicMock()
        instance.random_state = None

        args = ('some', 'args')
        kwargs = {'keyword': 'value'}

        random_mock.get_state.return_value = 'random state'

        # Run
        decorated_function = random_state(my_function)
        decorated_function(instance, *args, **kwargs)

        # Check
        my_function.assert_called_once_with(instance, *args, **kwargs)

        instance.assert_not_called
        random_mock.get_state.assert_not_called()
        random_mock.RandomState.assert_not_called()
        random_mock.set_state.assert_not_called()

    def test_validate_random_state_int(self):
        """Test `validate_random_state` with an int.

        Expect that the int is used to seed the RandomState object.

        Input:
            - integer seed
        Output:
            - np.Random.RandomState
        """
        # Setup
        state = 4

        # Run
        out = validate_random_state(state)

        # Assert
        assert isinstance(out, np.random.RandomState)

    def test_validate_random_state_none(self):
        """Test `validate_random_state` with an input of None.

        Expect that None is also returned.

        Input:
            - state of None
        Output:
            - None
        """
        # Setup
        state = None

        # Run
        validate_random_state(state)

        # Assert
        assert not state

    def test_validate_random_state_object(self):
        """Test `validate_random_state` with a `np.random.RandomState` object.

        Expect that the same object is returned.

        Input:
            - np.random.RandomState object
        Output:
            - state
        """
        # Setup
        state = np.random.RandomState(0)

        # Run
        out = validate_random_state(state)

        # Assert
        assert out == state

    def test_validate_random_state_invalid(self):
        """Test `validate_random_state` with an invalid input type.

        Expect a TypeError to be thrown.

        Input:
            - invalid input
        Side Effect:
            - TypeError
        """
        # Setup
        state = 'invalid input'

        # Run
        with pytest.raises(
                TypeError,
                match=f'`random_state` {state} expected to be an int or '
                '`np.random.RandomState` object.'):
            validate_random_state(state)


class TestGetInstance(TestCase):

    def test_get_instance_str(self):
        """Try to get a new instance from a str"""
        # Run
        instance = get_instance('copulas.multivariate.gaussian.GaussianMultivariate')

        # Asserts
        assert not instance.fitted
        assert isinstance(instance, GaussianMultivariate)

    def test_get_instance___class__(self):
        """Try to get a new instance from a __clas__"""
        # Run
        instance = get_instance(GaussianMultivariate)

        # Asserts
        assert not instance.fitted
        assert isinstance(instance, GaussianMultivariate)

    def test_get_instance_instance(self):
        """Try to get a new instance from a instance"""
        # Run
        instance = get_instance(GaussianMultivariate())

        # Asserts
        assert not instance.fitted
        assert isinstance(instance, GaussianMultivariate)

    def test_get_instance_instance_fitted(self):
        """Try to get a new instance from a fitted instance"""
        # Run
        gaussian = GaussianMultivariate()
        gaussian.fit(pd.DataFrame({'a_field': list(range(10))}))
        instance = get_instance(gaussian)

        # Asserts
        assert not instance.fitted
        assert isinstance(instance, GaussianMultivariate)

    def test_get_instance_instance_distribution(self):
        """Try to get a new instance from a instance with distribution"""
        # Run
        instance = get_instance(
            GaussianMultivariate(distribution='copulas.univariate.truncnorm.TruncNorm')
        )

        # Asserts
        assert not instance.fitted
        assert isinstance(instance, GaussianMultivariate)
        assert instance.distribution == 'copulas.univariate.truncnorm.TruncNorm'

    def test_get_instance_with_kwargs(self):
        """Try to get a new instance with kwargs"""
        # Run
        instance = get_instance(
            GaussianMultivariate,
            distribution='copulas.univariate.truncnorm.TruncNorm'
        )

        # Asserts
        assert not instance.fitted
        assert isinstance(instance, GaussianMultivariate)
        assert instance.distribution == 'copulas.univariate.truncnorm.TruncNorm'


@pytest.fixture()
def mock_copulas():
    copulas_module = sys.modules['copulas']
    copulas_mock = MagicMock()
    sys.modules['copulas'] = copulas_mock
    yield copulas_mock
    sys.modules['copulas'] = copulas_module


@patch.object(copulas, 'iter_entry_points')
def test__find_addons_module(entry_points_mock, mock_copulas):
    """Test loading an add-on."""
    # Setup
    entry_point = MagicMock()
    entry_point.name = 'copulas.submodule.entry_name'
    entry_point.load.return_value = 'entry_point'
    entry_points_mock.return_value = [entry_point]

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    assert mock_copulas.submodule.entry_name == 'entry_point'


@patch.object(copulas, 'iter_entry_points')
def test__find_addons_object(entry_points_mock, mock_copulas):
    """Test loading an add-on."""
    # Setup
    entry_point = MagicMock()
    entry_point.name = 'copulas.submodule:entry_object.entry_method'
    entry_point.load.return_value = 'new_method'
    entry_points_mock.return_value = [entry_point]

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    assert mock_copulas.submodule.entry_object.entry_method == 'new_method'


@patch('warnings.warn')
@patch('copulas.iter_entry_points')
def test__find_addons_bad_addon(entry_points_mock, warning_mock):
    """Test failing to load an add-on generates a warning."""
    # Setup
    def entry_point_error():
        raise ValueError()

    bad_entry_point = MagicMock()
    bad_entry_point.name = 'bad_entry_point'
    bad_entry_point.module_name = 'bad_module'
    bad_entry_point.load.side_effect = entry_point_error
    entry_points_mock.return_value = [bad_entry_point]
    msg = 'Failed to load "bad_entry_point" from "bad_module".'

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('copulas.iter_entry_points')
def test__find_addons_wrong_base(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = MagicMock()
    bad_entry_point.name = 'bad_base.bad_entry_point'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'bad_base.bad_entry_point': expected base module to be 'copulas', found "
        "'bad_base'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('copulas.iter_entry_points')
def test__find_addons_missing_submodule(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = MagicMock()
    bad_entry_point.name = 'copulas.missing_submodule.new_submodule'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'copulas.missing_submodule.new_submodule': module 'copulas' has no "
        "attribute 'missing_submodule'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('copulas.iter_entry_points')
def test__find_addons_module_and_object(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = MagicMock()
    bad_entry_point.name = 'copulas.missing_submodule:new_object'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'copulas.missing_submodule:new_object': cannot add 'new_object' to unknown "
        "submodule 'copulas.missing_submodule'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch.object(copulas, 'iter_entry_points')
def test__find_addons_missing_object(entry_points_mock, warning_mock, mock_copulas):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = MagicMock()
    bad_entry_point.name = 'copulas.submodule:missing_object.new_method'
    entry_points_mock.return_value = [bad_entry_point]
    msg = ("Failed to set 'copulas.submodule:missing_object.new_method': missing_object.")

    del mock_copulas.submodule.missing_object

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)
