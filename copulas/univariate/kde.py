from functools import partial

import numpy as np
import scipy

from copulas import scalarize
from copulas.univariate.base import Univariate


class KDEUnivariate(Univariate):
    """ A wrapper for gaussian Kernel density estimation implemented
    in scipy.stats toolbox. gaussian_kde is slower than statsmodels
    but allows more flexibility.
    """

    def __init__(self):
        super(KDEUnivariate, self).__init__()
        self.model = None

    def fit(self, X):
        """Fit Kernel density estimation to an array of values.

        Args:
            X(`np.ndarray` or `pd.DataFrame`):  Datapoints to be estimated from. Must be 1-d

        Returns:
            None
        """
        if not len(X):
            raise ValueError("data cannot be empty")

        self.constant_value = self._get_constant_value(X)

        if self.constant_value is None:
            self.model = scipy.stats.gaussian_kde(X)

        else:
            self._replace_constant_methods()

        self.fitted = True

    def probability_density(self, X):
        """Evaluate the estimated pdf on a point.

        Args:
            X(numpy.array): Points to evaluate its pdf.

        Returns:
            numpy.array: Value of estimated pdf for given points.
        """
        self.check_fit()

        return self.model.evaluate(X)

    def cumulative_distribution(self, X):
        """Computes the integral of a 1-D pdf between two bounds

        Args:
            X(numpy.array): Shaped (1, n), containing the datapoints.

        Returns:
            numpy.array: estimated cumulative distribution.
        """
        self.check_fit()

        low_bounds = self.model.dataset.mean() - (5 * self.model.dataset.std())

        result = []
        for value in X:
            result.append(self.model.integrate_box_1d(low_bounds, value))

        return np.array(result)

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
            value(float): cdf value, that is, in [0,1]

        Returns:
            callable: function whose zero is the ppf of value.
        """
        # The decorator expects an instance method, but usually are decorated before being bounded
        bound_cdf = partial(scalarize(KDEUnivariate.cumulative_distribution), self)

        def f(x):
            return bound_cdf(x) - value

        return f

    def percent_point(self, U):
        """Given a cdf value, returns a value in original space.

        Args:
            U(numpy.array): cdf values in [0,1]

        Returns:
            numpy.array: value in original space
        """
        self.check_fit()

        result = []
        for value in U:
            value = scipy.optimize.brentq(self._brentq_cdf(value), -1000.0, 1000.0)
            result.append(value)

        return np.array(result)

    def sample(self, num_samples=1):
        """Samples new data point based on model.

        Args:
            num_samples(int): number of points to be sampled

        Returns:
            samples: a list of datapoints sampled from the model
        """
        self.check_fit()

        return self.model.resample(num_samples)

    @classmethod
    def from_dict(cls, copula_dict):
        """Set attributes with provided values."""
        instance = cls()

        instance.fitted = copula_dict['fitted']
        instance.constant_value = copula_dict['constant_value']

        if instance.fitted and not instance.constant_value:
            instance.model = scipy.stats.gaussian_kde([-1, 0, 0])

            for key in ['dataset', 'covariance', 'inv_cov']:
                copula_dict[key] = np.array(copula_dict[key])

            attributes = ['d', 'n', 'dataset', 'covariance', 'factor', 'inv_cov']
            for name in attributes:
                setattr(instance.model, name, copula_dict[name])

        return instance

    def _fit_params(self):
        return {
            'd': self.model.d,
            'n': self.model.n,
            'dataset': self.model.dataset.tolist(),
            'covariance': self.model.covariance.tolist(),
            'factor': self.model.factor,
            'inv_cov': self.model.inv_cov.tolist()
        }
