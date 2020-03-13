import numpy as np
import pandas as pd

COMPARE_VALUES_ERROR = 'Values don\'t match at index {}\n {} != {}'
NUMPY_NUMERICAL_DTYPES = set('buifc')


def compare_nested_dicts(first, second, epsilon=10E-6):
    """Compares two dictionaries. Raises an assertion error when a difference is found."""

    assert first.keys() == second.keys()

    for key, _first in first.items():
        _second = second[key]
        if isinstance(_first, dict):
            compare_nested_dicts(_first, _second, epsilon)

        elif hasattr(_first, '__len__'):
            compare_nested_iterables(_first, _second, epsilon)

        elif isinstance(_first, pd.DataFrame):
            assert _first.equals(_second)

        elif isinstance(_first, float):
            message = COMPARE_VALUES_ERROR.format(key, _first, _second)
            assert compare_values_epsilon(_first, _second, epsilon), message

        else:
            assert _first == _second, "{} doesn't equal {}".format(_first, _second)


def compare_values_epsilon(first, second, epsilon=10E-6,):
    if pd.isnull(first) and pd.isnull(second):
        return True

    return abs(first - second) < epsilon


def compare_nested_iterables(first, second, epsilon=10E-6):

    assert len(first) == len(second), "Iterables should have the same length to be compared."

    for index, (_first, _second) in enumerate(zip(first, second)):

        message = COMPARE_VALUES_ERROR.format(index, _first, _second)

        if isinstance(_first, (list, tuple)):
            compare_nested_iterables(_first, _second, epsilon)

        elif isinstance(_first, np.ndarray):
            if _first.dtype.kind in NUMPY_NUMERICAL_DTYPES:
                np.testing.assert_allclose(_first, _second)
            else:
                compare_nested_iterables(_first, _second, epsilon)

        elif isinstance(_first, dict):
            compare_nested_dicts(_first, _second, epsilon)

        elif isinstance(_first, float):
            assert compare_values_epsilon(_first, _second, epsilon), message

        else:
            assert _first == _second, message


def copula_zero_if_arg_zero(copula, dimensions=2, steps=10, tolerance=1E-05):
    """Assert that any call with an argument equal to 0, will return 0.

    This function helps to test the following analytical property of copulas:
    If C is a copula and (u0, ..., uN) is in [0, 1]^N

    C(u0, ..., uM-1, 0, uM+1, ..., uN) = 0 for any 0 < M < N

    Args:
        copula(Bivariate or Multivariate): Copula instance to test.
        dimensions(int): Number of dimensions supported by given copulas.
        steps(int): Number of partitions of [0,1] to use to generate values.
        tolerance(float): Maximum difference in absolute value before raising error.

    Raises:
        AssertionError: If any value doesn't comply with the expected behavior.
    """
    # Setup
    step_values = np.linspace(0.0, 1.0, steps + 1)[1:]
    values = []

    for index in range(dimensions):
        for value in step_values:
            result = np.full(dimensions, value)
            result[index] = 0
            values.append(result.tolist())

    probabilities = np.array(values)
    expected_result = np.zeros(dimensions * steps)

    # Run
    result = copula.cdf(probabilities)

    # Check
    compare_nested_iterables(result, expected_result, tolerance)


def copula_single_arg_not_one(copula, dimensions=2, steps=10, tolerance=1E-05):
    """Assert that any call where all arguments minus one are 1, will return the non-1 value.

    This functions helps to test the following analytic property of copulas:
    If C is a copula and (u0, ..., uN) is in [0, 1]^N

    C(1, ..., 1, uM, 1, ..., 1) = uM for any 0 < M < N

    Args:
        copula(Bivariate or Multivariate): Copula instance to test
        dimensions(int): Number of dimensions supported by given copulas.
        steps(int): Number of partitions of [0,1] to use to generate values.
        tolerance(float): Maximum difference in absolute value before raising error.

    Raises:
        AssertionError: If any value doesn't comply with the expected behavior.
    """
    # Setup
    step_values = np.linspace(0.0, 1.0, steps + 1)[1: -1]
    values = []

    for index in range(dimensions):
        for value in step_values:
            result = np.ones(dimensions)
            result[index] = value
            values.append(result.tolist())

    probabilities = np.array(values)
    expected_result = np.tile(step_values, dimensions)

    # Run
    result = copula.cdf(probabilities)

    # Check
    compare_nested_iterables(result, expected_result, tolerance)
