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


def random_state(function):
    def wrapper(self, *args, **kwargs):
        if self.random_seed is None:
            return function(self, *args, **kwargs)

        else:
            original_state = np.random.get_state()
            np.random.seed(self.random_seed)

            result = function(self, *args, **kwargs)

            np.random.set_state(original_state)
            return result

    return wrapper


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
