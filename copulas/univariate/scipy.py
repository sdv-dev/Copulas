"""Scipy class that acts as a bridge between copulas and scipy."""

import importlib
import inspect


class ScipyDistribution:
    """Special class that wraps any given scipy distribution.

    This class is meant to be used with continuous distributions that are parametric and
    belong to ``scipy.stats`` module.

    Args:
        distribution (str):
            The python name of the distribution implemented in ``scipy.stats``. A complete list
            can be found at
            https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions .
            Defaults to ``'beta'``.

        fit_params (dict):
            A python dictionary with initialized fitting parameters for the given distribution.
            Defaults to ``None``
    """

    def __init__(self, distribution='beta', fit_params=None):
        self.distribution = getattr(importlib.import_module('scipy.stats'), distribution)
        self.fit_params = {} if fit_params is None else fit_params.copy()
        self._params = {}

    def fit(self, X):
        """Fit the model to a random variable.

        Args:
            X (numpy.ndarray):
                Values of the random variable. It must have shape (n, 1).
        """
        output = self.distribution.fit(X, **self.fit_params)
        args = inspect.getfullargspec(self.distribution._parse_args_rvs).args[1:-1]
        self._params = {}
        for idx, arg in enumerate(args):
            self._params[arg] = output[idx]

    def sample(self, n_samples=1):
        """Sample values from this model.

        Argument:
            n_samples (int):
                Number of values to sample

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, 1) with values randomly
                sampled from this model distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        return self.distribution.rvs(**self._params, size=n_samples)
