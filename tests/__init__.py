import numpy as np
import pandas as pd

COMPARE_VALUES_ERROR = 'Values don\'t match at index {}\n {} != {}'


def compare_nested_dicts(first, second, epsilon=10E-6):
    """Compares two dictionaries. Raises an assertion error when a difference is found."""

    assert first.keys() == second.keys()

    for key, _first in first.items():
        _second = second[key]
        if isinstance(_first, dict):
            compare_nested_dicts(_first, _second, epsilon)

        elif isinstance(_first, (list, np.ndarray, tuple)):
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

        if isinstance(_first, (list, np.ndarray, tuple)):
            compare_nested_iterables(_first, _second, epsilon)

        elif isinstance(_first, dict):
            compare_nested_dicts(_first, _second, epsilon)

        elif isinstance(_first, float):
            message = COMPARE_VALUES_ERROR.format(index, _first, _second)
            assert compare_values_epsilon(_first, _second, epsilon), message

        else:
            assert _first == _second, COMPARE_VALUES_ERROR.format(index, _first, _second)
