import numpy as np

from copulas import NotFittedError, get_qualified_name, import_object


class Univariate(object):
    """ Abstract class for representing univariate distributions """

    def __init__(self, random_seed=None):
        self.random_seed = random_seed
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

    @staticmethod
    def _get_constant_value(X):
        """Checks if a Series or array contains only one unique value.

        Args:
            X(pandas.Series or numpy.ndarray): Array to check for constantness

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
            numpy.ndarray: Sampled values. Array of shape (num_samples,).
        """
        return np.full(num_samples, self.constant_value)

    def _constant_cumulative_distribution(self, X):
        """Cumulative distribution for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are 0 and 1.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Args:
            X (numpy.ndarray): Values to compute cdf to.

        Returns:
            numpy.ndarray: Cumulative distribution for the given values.
        """
        result = np.ones(X.shape)
        result[np.nonzero(X < self.constant_value)] = 0

        return result

    def _constant_probability_density(self, X):
        """Probability density for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are 0 and 1.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Args:
            X(numpy.ndarray): Values to compute pdf.

        Returns:
            numpy.ndarray: Probability densisty for the given values
        """
        result = np.zeros(X.shape)
        result[np.nonzero(X == self.constant_value)] = 1

        return result

    def _constant_percent_point(self, X):
        """Percent point for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are `np.nan`
        and self.constant_value.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Args:
            X(numpy.ndarray): Percentiles.

        Returns:
            numpy.ndarray:

        """
        return np.full(X.shape, self.constant_value)

    def _replace_constant_methods(self):
        """Replaces conventional distribution methods by its constant counterparts."""
        self.cumulative_distribution = self._constant_cumulative_distribution
        self.percent_point = self._constant_percent_point
        self.probability_density = self._constant_probability_density
        self.sample = self._constant_sample
