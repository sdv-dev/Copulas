# -*- coding: utf-8 -*-

"""Top-level package for Copulas."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com',
__version__ = '0.2.2-dev'

import importlib

import numpy as np
import pandas as pd

EPSILON = np.finfo(np.float32).eps


class NotFittedError(Exception):
    pass


def import_object(object_name):
    """Import an object from its Fully Qualified Name."""
    package, name = object_name.rsplit('.', 1)
    return getattr(importlib.import_module(package), name)


def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class


def check_valid_values(function):
    """Raises an exception if the given values are not supported.

    Args:
        function(callable): Method whose unique argument is a numpy.array-like object.

    Returns:
        callable: Decorated function

    Raises:
        ValueError: If there are missing or invalid values or if the dataset is empty.
    """
    def decorated(self, X, *args, **kwargs):

        if isinstance(X, pd.DataFrame):
            W = X.values

        else:
            W = X

        if not len(W):
            raise ValueError('Your dataset is empty.')

        if W.dtype not in [np.dtype('float64'), np.dtype('int64')]:
            raise ValueError('There are non-numerical values in your data.')

        if np.isnan(W).any().any():
            raise ValueError('There are nan values in your data.')

        return function(self, X, *args, **kwargs)

    return decorated
