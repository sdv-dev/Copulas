import numpy as np
import scipy.stats

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
        """Return attributes from self.model to serialize.

        Returns:
            dict: Parameters to recreate self.model in its current fit status.
        """
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


class ScipyWrapper(Univariate):
    """Wrapper for scipy.stats.rv_continous subclasses.

    On fit time it will instantiate the given `model_class` name, and delete all of its
    methods that are not properly mapped to model's methods.

    If a method is not present on the model, but you want to implement it, you need to update
    the `method_map` for that method's to a non-None value and simply override the method.

    You can subclass `ScipyWrapper` by:
    1. On a file named after your distribution, create a new subclass.
    2. Set the `model_class` and `method_map` with the desired values for your model.
    3. Implement the `from_dict` and `_fit_params` methods.
    4. Implement any custom method not found in `model_class`.

    For a working example of how to implement subclasses of `ScipyWraper`, please check the source
    of `copulas.univariate.kde`.

    Class Attributes:
        model_class(str): Name of the model to use (Must be found in scipy.stats)
        method_map(dict): Mapping of the local names of methods to the name in the model.

    Args:
        None
    """

    model_class = None
    method_map = {
        'probability_density': None,
        'cumulative_distribution': None,
        'percent_point': None,
        'sample': None
    }

    def __init__(self):
        super(ScipyWrapper, self).__init__()
        self.model = None

        if not set(ScipyWrapper.method_map.keys()).issubset(set(self.method_map.keys())):
            raise ValueError(
                'This class hasn\'t been subclasses properly and some methods may crash')

        if not hasattr(scipy.stats, self.model_class):
            raise ValueError('The given `model_class` {} couldn\'t be found on scipy.stats')

    def fit(self, X):
        """Fit scipy model to an array of values.

        Args:
            X(`np.ndarray` or `pd.DataFrame`):  Datapoints to be estimated from. Must be 1-d

        Returns:
            None
        """

        self.constant_value = self._get_constant_value(X)

        if self.constant_value is None:
            self.model = getattr(scipy.stats, self.model_class)(X)
            for name, method_name in self.method_map.items():
                if method_name is None or not hasattr(self.model, method_name):
                    setattr(self, name, None)

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

        return getattr(self.model, self.method_map['probability_density'])(X)

    def cumulative_distribution(self, X):
        """Computes the integral of a 1-D pdf between two bounds

        Args:
            X(numpy.array): Shaped (1, n), containing the datapoints.

        Returns:
            numpy.array: estimated cumulative distribution.
        """
        self.check_fit()

        return getattr(self.model, self.method_map['cumulative_distribution'])(X)

    def percent_point(self, U):
        """Given a cdf value, returns a value in original space.

        Args:
            U(numpy.array): cdf values in [0,1]

        Returns:
            numpy.array: value in original space
        """
        self.check_fit()

        return getattr(self.model, self.method_map['percent_point'])(U)

    def sample(self, num_samples=1):
        """Samples new data point based on model.

        Args:
            num_samples(int): number of points to be sampled

        Returns:
            samples: a list of datapoints sampled from the model
        """
        self.check_fit()

        return getattr(self.model, self.method_map['sample'])(num_samples)

    @classmethod
    def from_dict(cls, parameters):
        """Set attributes with provided values.

        Args:
            parameters(dict): Dictionary containing instance parameters.

        Returns:
            ScipyWrapper: Instance populated with given parameters.
        """
        raise NotImplementedError
