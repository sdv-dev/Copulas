import numpy as np

from copulas import NotFittedError, get_qualified_name, import_object


class Univariate(object):
    """ Abstract class for representing univariate distributions """

    def __init__(self):
        self.fitted = False
        self.constant_value = None

    def fit(self, X):
        """Fits the model.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            None
        """
        raise NotImplementedError

    def probability_density(self, X):
        """Computes probability density.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray
        """
        raise NotImplementedError

    def pdf(self, X):
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Computes cumulative density.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray: Cumulative density for X.
        """
        raise NotImplementedError

    def cdf(self, X):
        return self.cumulative_distribution(X)

    def percent_point(self, U):
        """Given a cumulative distribution value, returns a value in original space.

        Arguments:
            U: `np.ndarray` of shape (n, 1) and values in [0,1]

        Returns:
            `np.ndarray`: Estimated values in original space.
        """
        raise NotImplementedError

    def ppf(self, U):
        return self.percent_point(U)

    def sample(self, n_samples=1):
        """Returns new data point based on model.

        Argument:
            n_samples: `int`

        Returns:
            np.ndarray: Generated samples
        """
        raise NotImplementedError

    def to_dict(self):
        """Returns parameters to replicate the distribution."""
        result = {
            'type': get_qualified_name(self),
            'fitted': self.fitted,
            'constant_value': self.constant_value
        }

        if not self.fitted:
            return result

        result.update(self._fit_params())
        return result

    def _fit_params(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, param_dict):
        """Create new instance from dictionary."""
        distribution_class = import_object(param_dict['type'])
        return distribution_class.from_dict(param_dict)

    def check_fit(self):
        """Assert that the object is fit

        Raises a `NotFittedError` if the model is  not fitted.
        """
        if not self.fitted:
            raise NotFittedError("This model is not fitted.")

    def check_constant_value(self):
        if self.constant_value:
            raise ValueError('This method is not available on constant distributions.')

    @staticmethod
    def _get_constant_value(X):
        """Checks if a Series or array contains only one unique value.

        Args:
            X(pandas.Series or numpy.array): Array to check for constantness

        Returns:
            (float or None): Return the constant value if there is one, else return None.
        """
        uniques = np.unique(X)
        if len(uniques) == 1:
            return uniques[0]

    def _constant_sample(self, num_samples):
        """Sample values for a constant distribution.

        Args:
            num_samples(int): Number of rows to sample

        Returns:
            numpy.array: Sampled values. Array of shape (num_samples,).
        """
        return np.array([self.constant_value] * num_samples)

    def _constant_cumulative_distribution(self, X):
        """Cumulative distribution for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are 0 and 1.

        Args:
            X (numpy.array): Values to compute cdf to.

        Returns:
            numpy.array: Cumulative distribution for the given values.
        """
        result = np.ones(X.shape)
        result[np.nonzero(X < self.constant_value)] = 0

        return result
