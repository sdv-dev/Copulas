from functools import partial

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import gaussian_kde

from copulas import EPSILON, scalarize, store_args
from copulas.univariate.base import BoundedType, ParametricType, ScipyModel


def _bisect(f, xmin, xmax, tol=1e-8, maxiter=50):
    a, b = xmin, xmax

    fa = f(a)
    fb = f(b)
    assert (fa <= 0.0).all()
    assert (fb >= 0.0).all()

    for _ in range(maxiter):
        c = (a + b) / 2.0  # proposal
        fc = f(c)
        a[fc <= 0] = c[fc <= 0]
        b[fc >= 0] = c[fc >= 0]
        if (b - a).max() < tol:
            break

    return (a + b) / 2.0


def _chandrupatla(f, xmin, xmax, verbose=False,
                  eps_m=None, eps_a=None,
                  maxiter=50, return_iter=False, args=(),):
    """
    This is adapted from [1] which implements Chandrupatla's algorithm [2]
    which starts from a bracketing interval and, conditionally, swaps between
    bisection and inverse quadratic interpolation.

    [1] https://github.com/scipy/scipy/issues/7242#issuecomment-290548427
    [2] https://books.google.com/books?id=cC-8BAAAQBAJ&pg=PA95
    """
    # Initialization
    a = xmax
    b = xmin
    fa = f(a, *args)
    fb = f(b, *args)

    # Make sure we know the size of the result
    shape = np.shape(fa)
    assert shape == np.shape(fb)

    fc = fa
    c = a

    # Make sure we are bracketing a root in each case
    assert (np.sign(fa) * np.sign(fb) <= 0).all()
    t = 0.5
    # Initialize an array of False,
    # determines whether we should do inverse quadratic interpolation
    iqi = np.zeros(shape, dtype=bool)

    # jms: some guesses for default values of the eps_m and eps_a settings
    # based on machine precision... not sure exactly what to do here
    eps = np.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2 * eps

    iterations = 0
    terminate = False

    while maxiter > 0:
        maxiter -= 1
        # use t to linearly interpolate between a and b,
        # and evaluate this function as our newest estimate xt
        xt = np.clip(a + t * (b - a), xmin, xmax)
        ft = f(xt, *args)
        if verbose:
            output = 'IQI? %s\nt=%s\nxt=%s\nft=%s\na=%s\nb=%s\nc=%s' % (iqi, t, xt, ft, a, b, c)
            print(output)
        # update our history of the last few points so that
        # - a is the newest estimate (we're going to update it from xt)
        # - c and b get the preceding two estimates
        # - a and b maintain opposite signs for f(a) and f(b)
        samesign = np.sign(ft) == np.sign(fa)
        c = np.choose(samesign, [b, a])
        b = np.choose(samesign, [a, b])
        fc = np.choose(samesign, [fb, fa])
        fb = np.choose(samesign, [fa, fb])
        a = xt
        fa = ft

        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        xm = np.choose(fa_is_smaller, [b, a])
        fm = np.choose(fa_is_smaller, [fb, fa])

        tol = 2 * eps_m * np.abs(xm) + eps_a
        tlim = tol / np.abs(b - c)
        terminate = np.logical_or(terminate, np.logical_or(fm == 0, tlim > 0.5))
        if verbose:
            output = "fm=%s\ntlim=%s\nterm=%s" % (fm, tlim, terminate)
            print(output)

        if np.all(terminate):
            break
        iterations += 1 - terminate

        # Figure out values xi and phi
        # to determine which method we should use next
        xi = (a - b) / (c - b)
        phi = (fa - fb) / (fc - fb)
        iqi = np.logical_and(phi**2 < xi, (1 - phi)**2 < 1 - xi)

        if not shape:
            # scalar case
            if iqi:
                # inverse quadratic interpolation
                t = fa / (fb - fa) * fc / (fb - fc) + (c - a) / \
                    (b - a) * fa / (fc - fa) * fb / (fc - fb)
            else:
                # bisection
                t = 0.5
        else:
            # array case
            t = np.full(shape, 0.5)
            a2, b2, c2, fa2, fb2, fc2 = a[iqi], b[iqi], c[iqi], fa[iqi], fb[iqi], fc[iqi]
            t[iqi] = fa2 / (fb2 - fa2) * fc2 / (fb2 - fc2) + (c2 - a2) / \
                (b2 - a2) * fa2 / (fc2 - fa2) * fb2 / (fc2 - fb2)

        # limit to the range (tlim, 1-tlim)
        t = np.minimum(1 - tlim, np.maximum(tlim, t))

    # done!
    if return_iter:
        return xm, iterations
    else:
        return xm


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
    def __init__(self, sample_size=None, random_seed=None, bw_method=None, weights=None):
        self.random_seed = random_seed
        self._sample_size = sample_size
        self.bw_method = bw_method
        self.weights = weights

    def _get_model(self):
        dataset = self._params['dataset']
        self._sample_size = self._sample_size or len(dataset)
        return gaussian_kde(dataset, bw_method=self.bw_method, weights=self.weights)

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
        X = np.array(X)
        stdev = np.sqrt(self._model.covariance[0, 0])
        lower = ndtr((self._get_bounds()[0] - self._model.dataset) / stdev)[0]
        uppers = ndtr((X[:, None] - self._model.dataset) / stdev)
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

    def percent_point_slow(self, U):
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
                    (self.percent_point_slow(u[0]) for u in U),
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

    def percent_point(self, U, method="bisect"):
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

        if len(U.shape) > 1:
            raise ValueError("Expected 1d array, got %s." % (U, ))

        if np.any(U > 1.0) or np.any(U < 0.0):
            raise ValueError("Expected values in range [0.0, 1.0].")

        is_one = U >= 1.0 - EPSILON
        is_zero = U <= EPSILON
        is_valid = ~(is_zero | is_one)

        lower, upper = self._get_bounds()

        def _f(X):
            return self.cumulative_distribution(X) - U[is_valid]

        X = np.zeros(U.shape)
        X[is_one] = float("inf")
        X[is_zero] = float("-inf")
        if is_valid.any():
            lower = np.full(U[is_valid].shape, lower)
            upper = np.full(U[is_valid].shape, upper)
            if method == "bisect":
                X[is_valid] = _bisect(_f, lower, upper)
            else:
                X[is_valid] = _chandrupatla(_f, lower, upper)

        return X

    def _fit_constant(self, X):
        sample_size = self._sample_size or len(X)
        constant = np.unique(X)[0]
        self._params = {
            'dataset': [constant] * sample_size,
        }

    def _fit(self, X):
        if self._sample_size:
            X = gaussian_kde(X, bw_method=self.bw_method,
                             weights=self.weights).resample(self._sample_size)
        self._params = {
            'dataset': X.tolist()
        }

    def _is_constant(self):
        return len(np.unique(self._params['dataset'])) == 1
