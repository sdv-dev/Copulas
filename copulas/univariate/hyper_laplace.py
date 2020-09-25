from random import random

import numpy as np
from scipy.special import gamma, psi
import scipy.stats

from copulas.univariate.base import BoundedType, ParametricType, Univariate


def gamma_d(x):
    """
    Compute the derivative of gamma function

    Arguments:
            x (float)

        Returns:
            float:
                The derivative of gamma function at x
    """
    return gamma(x) * psi(x)


def g3g1_over_g22(x):
    """
    Commute the value of Gamma(3x)Gamma(x)/Gamma^2(2x)

    Arguments:
            x (float)

        Returns:
            float:
                The value of Gamma(3x)Gamma(x)/Gamma^2(2x)
    """
    return gamma(3 * x) * gamma(x) / gamma(2 * x) ** 2


def g3g1_over_g22_d(x):
    """
    Compute the derivative of Gamma(3x)Gamma(x)/Gamma^2(2x)

    Arguments:
            x (float)

        Returns:
            float:
                The derivative of Gamma(3x)Gamma(x)/Gamma^2(2x) at x
    """
    x1 = gamma(x)
    x2 = gamma(2 * x)
    x3 = gamma(3 * x)

    x3_d = 3 * gamma_d(3 * x) * x1 * (x2 ** 2)
    x1_d = x3 * gamma_d(x) * (x2 ** 2)
    x2_d = 4 * x3 * x1 * gamma_d(2 * x) * x2
    return (x3_d + x1_d - x2_d) / (x2 ** 4)


def solve_gamma_equation(y, num_iter=10, upper_threshold=2, lower_threshold=0.5):
    """
    Solve for the equation Gamma(3x)Gamma(x)/Gamma^2(2x) = y using Newton's method

    Arguments:
            y (float)

            num_iter (int):
                number of iterations when using Newton's method

            upper_threshold (float):
                Maximum value of x_new/x_old allowed in Newton's method. Should be > 1

            lower_threshold (float):
                Minimum value of x_new/x_old allowed in Newton's method. Should be in (0,1)

        Returns:
            float:
                The solution to Gamma(3x)Gamma(x)/Gamma^2(2x) = y
    """
    x = 1
    for i in range(num_iter):
        x_new = x + (y - g3g1_over_g22(x)) / g3g1_over_g22_d(x)
        if x_new > upper_threshold * x:
            x = upper_threshold * x
        elif x_new < lower_threshold * x:
            x = lower_threshold * x
        else:
            x = x_new
    return x


class HyperLaplace(Univariate):
    """
    An HyperLaplace model object, implemented via a wrapper around scipy.stats.gamma

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

    Math derivation: HyperLaplace(k, alpha) =
    (gamma(loc = 0, scale = 1/k, a = 1/alpha))**(1/alpha) * Unif({-1,1})
    """

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    MODEL_CLASS = scipy.stats.gamma

    _params = None
    _model = None

    alpha = None
    k = None

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
        return self._model.pdf(abs(X) ** self.alpha) / 2 * self.alpha * abs(X) ** (self.alpha - 1)

    def log_probability_density(self, X):
        """Compute the log of the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the log probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Log probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()

        return np.log(self.probability_density(X))

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
        gamma_cdf = self._model.cdf(abs(X) ** self.alpha)
        for i in range(len(X)):
            cd, x = gamma_cdf[i], X[i]
            if x > 0:
                gamma_cdf[i] = (cd + 1) / 2
            else:
                gamma_cdf[i] = (1 - cd) / 2

        return gamma_cdf

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
        gamma_ppf = self._model.ppf(abs(U - 0.5) * 2) ** (1 / self.alpha)
        for i in range(len(U)):
            pp, u = gamma_ppf[i], U[i]
            if u < 0.5:
                gamma_ppf[i] = - pp

        return gamma_ppf

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
        gamma_rvs = self._model.rvs(n_samples) ** (1 / self.alpha)
        for i in range(n_samples):
            if random() > 0.5:
                gamma_rvs[i] = - gamma_rvs[i]
        return gamma_rvs

    def _fit(self, X):
        """Fit the model to a non-constant random variable.

        This fitting method is implemented by matching the theoritical mean
        and variance with the emperical mean and variance.

        Arguments:
            X (numpy.ndarray):
                Values of the random variable. It must have shape (n, 1).
        """
        mean = np.mean(abs(X))
        square = np.mean(X ** 2)
        a = solve_gamma_equation(square / (mean ** 2))
        self.alpha = 1 / a
        self.k = (gamma(2 * a) / (gamma(a) * mean)) ** self.alpha

        self._params = {
            'loc': 0,
            'scale': 1 / self.k,
            'a': a
        }

    def _get_model(self):
        return self.MODEL_CLASS(**self._params)

    def _fit_constant(self, X):
        self._params = {
            'loc': np.unique(X)[0],
            'scale': 0,
            'a': 1
        }

    def _is_constant(self):
        return self._params['scale'] == 0

    def fit(self, X):
        """Fit the model to a random variable.

        Arguments:
            X (numpy.ndarray):
                Values of the random variable. It must have shape (n, 1).
        """
        if self._check_constant_value(X):
            self._fit_constant(X)
        else:
            self._fit(X)
            self._model = self._get_model()

        self.fitted = True

    def _get_params(self):
        """Return attributes from self._model to serialize.

        Must be implemented in all the subclasses.

        Returns:
            dict:
                Parameters to recreate self._model in its current fit status.
        """
        return self._params.copy()

    def _set_params(self, params):
        """Set the parameters of this univariate.

        Args:
            params (dict):
                Parameters to recreate this instance.
        """
        self._params = params.copy()
        self.k = 1 / self._params['scale']
        self.alpha = 1 / self._params['a']
        if self._is_constant():
            self._replace_constant_methods()
        else:
            self._model = self._get_model()
