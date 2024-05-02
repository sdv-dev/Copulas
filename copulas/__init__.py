# -*- coding: utf-8 -*-

"""Top-level package for Copulas."""

__author__ = 'DataCebo, Inc.'
__email__ = 'info@sdv.dev'
__version__ = '0.11.1.dev0'

import contextlib
import importlib
import sys
import warnings
from copy import deepcopy
from importlib.metadata import entry_points
from operator import attrgetter

import numpy as np
import pandas as pd

EPSILON = np.finfo(np.float32).eps


class NotFittedError(Exception):
    """NotFittedError class."""


@contextlib.contextmanager
def set_random_state(random_state, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_state (int or np.random.RandomState):
            The random seed or RandomState.
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_state = np.random.get_state()

    np.random.set_state(random_state.get_state())

    try:
        yield
    finally:
        current_random_state = np.random.RandomState()
        current_random_state.set_state(np.random.get_state())
        set_model_random_state(current_random_state)
        np.random.set_state(original_state)


def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    def wrapper(self, *args, **kwargs):
        if self.random_state is None:
            return function(self, *args, **kwargs)

        else:
            with set_random_state(self.random_state, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper


def validate_random_state(random_state):
    """Validate random state argument.

    Args:
        random_state (int, numpy.random.RandomState, tuple, or None):
            Seed or RandomState for the random generator.

    Output:
        numpy.random.RandomState
    """
    if random_state is None:
        return None

    if isinstance(random_state, int):
        return np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise TypeError(
            f'`random_state` {random_state} expected to be an int '
            'or `np.random.RandomState` object.'
        )


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
        if kwargs:
            instance = obj.__class__(**kwargs)
        else:
            args = getattr(obj, '__args__', ())
            kwargs = getattr(obj, '__kwargs__', {})
            instance = obj.__class__(*args, **kwargs)

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
                (function(self, *x, *args, **kwargs) for x in X), np.dtype('float64')
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
    """Raise an exception if the given values are not supported.

    Args:
        function(callable): Method whose unique argument is a numpy.array-like object.

    Returns:
        callable: Decorated function

    Raises:
        ValueError: If there are missing or invalid values or if the dataset is empty.
    """

    def decorated(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            W = X.to_numpy()

        else:
            W = X

        if not len(W):
            raise ValueError('Your dataset is empty.')

        if not (np.issubdtype(W.dtype, np.floating) or np.issubdtype(W.dtype, np.integer)):
            raise ValueError('There are non-numerical values in your data.')

        if np.isnan(W).any().any():
            raise ValueError('There are nan values in your data.')

        return function(self, X, *args, **kwargs)

    return decorated


def _get_addon_target(addon_path_name):
    """Find the target object for the add-on.

    Args:
        addon_path_name (str):
            The add-on's name. The add-on's name should be the full path of valid Python
            identifiers (i.e. importable.module:object.attr).

    Returns:
        tuple:
            * object:
                The base module or object the add-on should be added to.
            * str:
                The name the add-on should be added to under the module or object.
    """
    module_path, _, object_path = addon_path_name.partition(':')
    module_path = module_path.split('.')

    if module_path[0] != __name__:
        msg = f"expected base module to be '{__name__}', found '{module_path[0]}'"
        raise AttributeError(msg)

    target_base = sys.modules[__name__]
    for submodule in module_path[1:-1]:
        target_base = getattr(target_base, submodule)

    addon_name = module_path[-1]
    if object_path:
        if len(module_path) > 1 and not hasattr(target_base, module_path[-1]):
            msg = f"cannot add '{object_path}' to unknown submodule '{'.'.join(module_path)}'"
            raise AttributeError(msg)

        if len(module_path) > 1:
            target_base = getattr(target_base, module_path[-1])

        split_object = object_path.split('.')
        addon_name = split_object[-1]

        if len(split_object) > 1:
            target_base = attrgetter('.'.join(split_object[:-1]))(target_base)

    return target_base, addon_name


def _find_addons():
    """Find and load all copulas add-ons."""
    group = 'copulas_modules'
    try:
        eps = entry_points(group=group)
    except TypeError:
        # Load-time selection requires Python >= 3.10 or importlib_metadata >= 3.6
        eps = entry_points().get(group, [])

    for entry_point in eps:
        try:
            addon = entry_point.load()
        except Exception:  # pylint: disable=broad-exception-caught
            msg = f'Failed to load "{entry_point.name}" from "{entry_point.value}".'
            warnings.warn(msg)
            continue

        try:
            addon_target, addon_name = _get_addon_target(entry_point.name)
        except AttributeError as error:
            msg = f"Failed to set '{entry_point.name}': {error}."
            warnings.warn(msg)
            continue

        setattr(addon_target, addon_name, addon)


_find_addons()
