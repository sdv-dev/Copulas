import numpy as np
import scipy.stats

from copulas import NotFittedError, check_valid_values, get_instance, get_qualified_name


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
        distribution_class = get_instance(param_dict['type'])
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
    """Wrapper for :attr:`scipy.stats.rv_continous` subclasses.

    This class is intended to be used to integrate random variables from :attr:`scipy.stats`
    into copulas. It contain 5 attributes that control its behavior:

    - :attr:`model_class`: Name of the class to integrate, it must be in the :attr:`scipy.stats`
      module.
    - :attr:`probability_density`, :attr:`cumulative_distribution`, :attr:`percent_point`,
      :attr:`sample` : This attributes contain the information about how to map the methods in the
      model.

      * If it's of type :attr:`str` it will interpreted as the name of the corresponding method in
        model.

      * If it's :attr:`None` it will be interpreted as the method doesn't exist in the model and
        is not implemented and a error message will be displayed.

      * If it's not present it will be interpreted as the method doesn't exist in the model but
        has been implemented in the integration.

    On fit time it will instantiate the given :attr:`model_class` name, and map its methods to the
    class, following the values in the attributes for the corresponding method.

    You can subclass :attr:`ScipyWrapper` by:

    1. On a file named after your distribution, create a new subclass.

    2. Set the :attr:`model_class` with the name of the distribution you want to integrate.

    3. Map the methods of your model.

    4. Implement the :attr:`from_dict` and :attr:`_fit_params` methods.

    5. Implement any custom method not mapped.

    For a working example of how to implement subclasses of :attr:`ScipyWraper`, please check the
    source of :attr:`copulas.univariate.kde`.

    Attributes:
        model(scipy.stats.rv_continuous): Actual scipy.stats instance we are wrapping.
        model_class(str): Name of the model to use (Must be found in scipy.stats)
        probability_density(str): Name of the method of model to map to :attr:`probability_density`
        percent_point(str): Name of the method of model to map to :attr:`percent_point`
        sample(str): Name of the method of model to map to :attr:`sample`
        cumulative_distribution(str):
            Name of the method of model to map to :attr:`cumulative_distribution`
        unfittable_model(bool):
            Wheter or not if the wrapper method needs data to be created or only parameters.
            (Examaples of both behaviors are :attr:`GaussianKDE` and :attr:`TruncNorm`)
    Args:
        None
    """

    model = None
    model_class = None
    unfittable_model = None
    probability_density = None
    cumulative_distribution = None
    percent_point = None
    sample = None
    METHOD_NAMES = ('sample', 'probability_density', 'cumulative_distribution', 'percent_point')

    def __init__(self, *args, **kwargs):
        super(ScipyWrapper, self).__init__(*args, **kwargs)

    def _replace_methods(self):
        for name in self.METHOD_NAMES:
            attribute = getattr(self.__class__, name)
            if isinstance(attribute, str):
                setattr(self, name, getattr(self.model, attribute))

    @check_valid_values
    def fit(self, X, *args, **kwargs):
        """Fit scipy model to an array of values.

        Args:
            X(`np.ndarray` or `pd.DataFrame`):  Datapoints to be estimated from. Must be 1-d

        Returns:
            None
        """

        self.constant_value = self._get_constant_value(X)

        if self.constant_value is None:
            if self.unfittable_model:
                self.model = getattr(scipy.stats, self.model_class)(*args, **kwargs)
            else:
                self.model = getattr(scipy.stats, self.model_class)(X, *args, **kwargs)

            self._replace_methods()

        else:
            self._replace_constant_methods()

        self.fitted = True

    @classmethod
    def from_dict(cls, parameters):
        """Set attributes with provided values.

        Args:
            parameters(dict): Dictionary containing instance parameters.

        Returns:
            ScipyWrapper: Instance populated with given parameters.
        """
        raise NotImplementedError
