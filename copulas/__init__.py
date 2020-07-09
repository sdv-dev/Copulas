# -*- coding: utf-8 -*-

"""Top-level package for Copulas."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com',
__version__ = '0.3.2.dev0'

import contextlib
import importlib
from copy import deepcopy

import numpy as np
import pandas as pd

EPSILON = np.finfo(np.float32).eps


class NotFittedError(Exception):
    pass


@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def random_state(function):
    def wrapper(self, *args, **kwargs):
        if self.random_seed is None:
            return function(self, *args, **kwargs)

        else:
            with random_seed(self.random_seed):
                return function(self, *args, **kwargs)

    return wrapper


def get_instance(obj, **kwargs):
    """Create new instance of the ``obj`` argument.

    Args:
        obj (str, type, instance):
    """
    instance = None
    if isinstance(obj, str):
        package, name = obj.rsplit('.', 1)
        instance = getattr(importlib.import_module(package), name)(**kwargs)

    elif isinstance(obj, type):
        instance = obj(**kwargs)

    else:
        if kwargs != dict():
            instance = obj.__class__(**kwargs)

        else:
            instance = obj.__class__(*obj.__args__, **obj.__kwargs__)

    return instance


def store_args(__init__):
    """Save ``*args`` and ``**kwargs`` used in the ``__init__`` of a copula.

    Args:
        __init__(callable): ``__init__`` function to store their arguments.

    Returns:
        callable: Decorated ``__init__`` function.
    """

    def new__init__(self, *args, **kwargs):
        args_copy = deepcopy(args)
        kwargs_copy = deepcopy(kwargs)
        __init__(self, *args, **kwargs)
        self.__args__ = args_copy
        self.__kwargs__ = kwargs_copy

    return new__init__


def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class


def vectorize(function):
    """Allow a method that only accepts scalars to accept vectors too.

    This decorator has two different behaviors depending on the dimensionality of the
    array passed as an argument:

    **1-d array**

    It will work under the assumption that the `function` argument is a callable
    with signature::

        function(self, X, *args, **kwargs)

    where X is an scalar magnitude.

    In this case the arguments of the input array will be given one at a time, and
    both the input and output of the decorated function will have shape (n,).

    **2-d array**

    It will work under the assumption that the `function` argument is a callable with signature::

        function(self, X0, ..., Xj, *args, **kwargs)

    where `Xi` are scalar magnitudes.

    It will pass the contents of each row unpacked on each call. The input is espected to have
    shape (n, j), the output a shape of (n,)

    It will return a function that is guaranteed to return a `numpy.array`.

    Args:
        function(callable): Function that only accept and return scalars.

    Returns:
        callable: Decorated function that can accept and return :attr:`numpy.array`.

    """

    def decorated(self, X, *args, **kwargs):
        if not isinstance(X, np.ndarray):
            return function(self, X, *args, **kwargs)

        if len(X.shape) == 1:
            X = X.reshape([-1, 1])

        if len(X.shape) == 2:
            return np.fromiter(
                (function(self, *x, *args, **kwargs) for x in X),
                np.dtype('float64')
            )

        else:
            raise ValueError('Arrays of dimensionality higher than 2 are not supported.')

    decorated.__doc__ = function.__doc__
    return decorated


def scalarize(function):
    """Allow methods that only accepts 1-d vectors to work with scalars.

    Args:
        function(callable): Function that accepts and returns vectors.

    Returns:
        callable: Decorated function that accepts and returns scalars.
    """

    def decorated(self, X, *args, **kwargs):
        scalar = not isinstance(X, np.ndarray)

        if scalar:
            X = np.array([X])

        result = function(self, X, *args, **kwargs)
        if scalar:
            result = result[0]

        return result

    decorated.__doc__ = function.__doc__
    return decorated


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
