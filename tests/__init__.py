import numpy as np
import pandas as pd


def compare_nested_dicts(first, second, epsilon=10E-6):
    """Compares two dictionaries. Raises an assertion error when a difference is found."""

    assert first.keys() == second.keys()

    for key in first.keys():
        if isinstance(first[key], dict):
            compare_nested_dicts(first[key], second[key])

        elif isinstance(first[key], np.ndarray):
            assert (compare_values_epsilon(first[key], second[key])).all()

        elif isinstance(first[key], pd.DataFrame):
            assert first[key].equals(second[key])

        elif isinstance(first[key], float):
            assert compare_values_epsilon(first[key], second[key])

        elif isinstance(first[key], list):
            compare_nested_iterables(first[key], second[key])

        else:
            assert first[key] == second[key], "{} doesn't equal {}".format(first[key], second[key])


def compare_values_epsilon(first, second, epsilon=10E-6):
    return abs(first - second) < epsilon


def compare_nested_iterables(first, second, epsilon=10E-6):

    for _first, _second in zip(first, second):

        if isinstance(_first, list):
            compare_nested_iterables(_first, _second)

        if isinstance(_first, float):
            assert compare_values_epsilon(_first, _second)
