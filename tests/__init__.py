import numpy as np
import pandas as pd

COMPARE_VALUES_ERROR = 'Values don\'t match at index {}\n {} != {}'


def compare_nested_dicts(first, second, epsilon=10E-6):
    """Compares two dictionaries. Raises an assertion error when a difference is found."""

    assert first.keys() == second.keys()

    for key in first.keys():
        if isinstance(first[key], dict):
            compare_nested_dicts(first[key], second[key])

        elif isinstance(first[key], (list, np.ndarray, tuple)):
            compare_nested_iterables(first[key], second[key])

        elif isinstance(first[key], pd.DataFrame):
            assert first[key].equals(second[key])

        elif isinstance(first[key], float):
            assert compare_values_epsilon(first[key], second[key], key)

        else:
            assert first[key] == second[key], "{} doesn't equal {}".format(first[key], second[key])


def compare_values_epsilon(first, second, index=None, epsilon=10E-6,):
    return abs(first - second) < epsilon, COMPARE_VALUES_ERROR.format(index, first, second)


def compare_nested_iterables(first, second, epsilon=10E-6):

    assert len(first) == len(second), "Iterables should have the same length to be compared."

    for index, (_first, _second) in enumerate(zip(first, second)):

        if isinstance(_first, (list, np.ndarray, tuple)):
            compare_nested_iterables(_first, _second)

        elif isinstance(_first, dict):
            compare_nested_dicts(_first, _second)

        elif isinstance(_first, float):
            assert compare_values_epsilon(_first, _second)

        else:
            assert _first == _second, COMPARE_VALUES_ERROR.format(index, _first, _second)
