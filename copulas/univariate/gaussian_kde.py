from functools import partial

import numpy as np
import scipy

from copulas import scalarize, store_args, vectorize
from copulas.univariate.base import ScipyWrapper


class GaussianKDE(ScipyWrapper):
    """A wrapper for gaussian Kernel density estimation implemented
    in scipy.stats toolbox. gaussian_kde is slower than statsmodels
    but allows more flexibility.

    When a sample_size is provided the fit method will sample the
    data, and mask the real information. Also, ensure the number of
    entries will be always the value of sample_size.

    Args:
        sample_size(int): amount of parameters to sample
    """

    model_class = 'gaussian_kde'
    probability_density = 'evaluate'
    sample = 'resample'

    @store_args
    def __init__(self, sample_size=None, *args, **kwargs):
        self.sample_size = sample_size
        super().__init__(*args, **kwargs)

    def fit(self, X, *args, **kwargs):
        self.constant_value = self._get_constant_value(X)

        if self.constant_value is None:
            if self.sample_size:
                X = self.sample(self.sample_size)
                super().fit(X, *args, **kwargs)

            super().fit(X, *args, **kwargs)

        else:
            self._replace_constant_methods()

        self.fitted = True

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
        bound_cdf = partial(scalarize(GaussianKDE.cumulative_distribution), self)

        def f(x):
            return bound_cdf(x) - value

        return f

    @vectorize
    def percent_point(self, U):
        """Given a cdf value, returns a value in original space.

        Args:
            U(numpy.array): cdf values in [0,1]

        Returns:
            numpy.array: value in original space
        """
        self.check_fit()

        return scipy.optimize.brentq(self._brentq_cdf(U), -1000.0, 1000.0)

    @classmethod
    def from_dict(cls, copula_dict):
        """Set attributes with provided values."""
        instance = cls()

        instance.fitted = copula_dict['fitted']

        if instance.fitted:
            X = np.array(copula_dict['dataset'])
            uniques = np.unique(X)
            if len(uniques) == 1:
                instance.constant_value = uniques[0]

            else:
                instance.model = scipy.stats.gaussian_kde(X)

        return instance

    def _fit_params(self):
        if self.constant_value is not None:
            return {
                'dataset': [self.constant_value] * self.sample_size,
            }

        return {
            'dataset': self.model.dataset.tolist(),
        }
