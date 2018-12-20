# -*- coding: utf-8 -*-

"""Top-level package for Copulas."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com',
__version__ = '0.2.1-dev'

import numpy as np

EPSILON = np.finfo(np.float32).eps


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
