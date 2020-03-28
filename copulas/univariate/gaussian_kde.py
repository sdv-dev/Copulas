from functools import partial

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import gaussian_kde

from copulas import EPSILON, scalarize, store_args
from copulas.univariate.base import BoundedType, ParametricType, ScipyModel


class GaussianKDE(ScipyModel):
    """A wrapper for gaussian Kernel density estimation implemented
    in scipy.stats toolbox. gaussian_kde is slower than statsmodels
    but allows more flexibility.

    When a sample_size is provided the fit method will sample the
    data, and mask the real information. Also, ensure the number of
    entries will be always the value of sample_size.

    Args:
        sample_size(int): amount of parameters to sample
    """

    PARAMETRIC = ParametricType.NON_PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED
    MODEL_CLASS = gaussian_kde

    @store_args
    def __init__(self, sample_size=None, random_seed=None):
        self.random_seed = random_seed
        self._sample_size = sample_size

    def _get_model(self):
        dataset = self._params['dataset']
        self._sample_size = self._sample_size or len(dataset)
        return gaussian_kde(dataset)

    def _get_bounds(self):
        X = self._params['dataset']
        lower = np.min(X) - (5 * np.std(X))
        upper = np.max(X) + (5 * np.std(X))

        return lower, upper

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._model.evaluate(X)

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
        self.check_fit()
        return self._model.resample(size=n_samples)[0]

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        stdev = np.sqrt(self._model.covariance[0, 0])
        lower = ndtr((self._get_bounds()[0] - self._model.dataset) / stdev)[0]
        uppers = np.vstack([ndtr((x - self._model.dataset) / stdev)[0] for x in X])
        return (uppers - lower).dot(self._model.weights)

    def _brentq_cdf(self, value):
        """Helper function to compute percent_point.

        As scipy.stats.gaussian_kde doesn't provide this functionality out of the box we need
        to make a numerical approach:

        - First we scalarize and bound cumulative_distribution.
        - Then we define a function `f(x) = cdf(x) - value`, where value is the given argument.
        - As value will be called from ppf we can assume value = cdf(z) for some z that is the
        value we are searching for. Therefore the zeros of the function will be x such that:
        cdf(x) - cdf(z) = 0 => (becasue cdf is monotonous and continous) x = z

        Args:
            value (float):
                cdf value, that is, in [0,1]

        Returns:
            callable:
                function whose zero is the ppf of value.
        """
        # The decorator expects an instance method, but usually are decorated before being bounded
        bound_cdf = partial(scalarize(GaussianKDE.cumulative_distribution), self)

        def f(x):
            return bound_cdf(x) - value

        return f

    def percent_point(self, U):
        """Compute the inverse cumulative distribution value for each point in U.

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()

        if isinstance(U, np.ndarray):
            if len(U.shape) == 1:
                U = U.reshape([-1, 1])

            if len(U.shape) == 2:
                return np.fromiter(
                    (self.percent_point(u[0]) for u in U),
                    np.dtype('float64')
                )

            else:
                raise ValueError('Arrays of dimensionality higher than 2 are not supported.')

        if np.any(U > 1.0) or np.any(U < 0.0):
            raise ValueError("Expected values in range [0.0, 1.0].")

        is_one = U >= 1.0 - EPSILON
        is_zero = U <= EPSILON
        is_valid = not (is_zero or is_one)

        lower, upper = self._get_bounds()

        X = np.zeros(U.shape)
        X[is_one] = float("inf")
        X[is_zero] = float("-inf")
        X[is_valid] = brentq(self._brentq_cdf(U[is_valid]), lower, upper)

        return X

    def _fit_constant(self, X):
        sample_size = self._sample_size or len(X)
        constant = np.unique(X)[0]
        self._params = {
            'dataset': [constant] * sample_size,
        }

    def _fit(self, X):
        if self._sample_size:
            X = gaussian_kde(X).resample(self._sample_size)

        self._params = {
            'dataset': X.tolist()
        }

    def _is_constant(self):
        return len(np.unique(self._params['dataset'])) == 1
