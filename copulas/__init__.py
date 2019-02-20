# -*- coding: utf-8 -*-

"""Top-level package for Copulas."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com',
__version__ = '0.2.2-dev'

import importlib

import numpy as np

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


def vectorize(function):
    """Allow a methods that only accepts scalars to work with vectors.

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
        function(callable): Function that accepts and returns scalars.

    Returns:
        callable: Decorated function that accepts and returns `numpy.array`.
    """

    def decorated(self, X, *args, **kwargs):
        if len(X.shape) == 1:
            return np.fromiter(
                (function(self, x, *args, **kwargs) for x in X),
                np.dtype('float64')
            )

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
        return function(self, np.array([X]), *args, **kwargs)[0]

    decorated.__doc__ = function.__doc__
    return decorated
