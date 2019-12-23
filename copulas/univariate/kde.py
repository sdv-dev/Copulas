import numpy as np
import scipy

from copulas import check_valid_values, random_state, store_args
from copulas.univariate.base import Univariate


class KDEUnivariate(Univariate):
    """ A wrapper for gaussian Kernel density estimation implemented
    in scipy.stats toolbox. gaussian_kde is slower than statsmodels
    but allows more flexibility.
    """

    @store_args
    def __init__(self, sample_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
        self.model = None

    @check_valid_values
    def fit(self, X):
        """Fit Kernel density estimation to an list of values.

        Args:
            X: 1-d `np.ndarray` or `pd.Series` or `list` datapoints to be estimated from.

        This function will fit a gaussian_kde model to a list of datapoints
        and store it as a class attribute.
        """

        self.constant_value = self._get_constant_value(X)

        if self.constant_value is None:
            if self.sample_size is not None:
                model = scipy.stats.gaussian_kde(X)
                X = model.resample(self.sample_size)

            self.model = scipy.stats.gaussian_kde(X)

        else:
            self._replace_constant_methods()

        self.fitted = True

    def probability_density(self, X):
        """Evaluate the estimated pdf on a point.

        Args:
            X(float):  a datapoint.

        Returns:
            (float): value of estimated pdf.

        """
        self.check_fit()

        if type(X) not in (int, float):
            raise ValueError('x must be int or float')

        return self.model.evaluate(X)[0]

    def cumulative_distribution(self, X, U=0):
        """Computes the integral of a 1-D pdf between two bounds

        Args:
            X(float): a datapoint.
            U(float): cdf value in [0,1], only used in get_ppf

        Returns:
            float: estimated cumulative distribution.
        """
        self.check_fit()

        low_bounds = self.model.dataset.mean() - (5 * self.model.dataset.std())
        return self.model.integrate_box_1d(low_bounds, X) - U

    def percent_point(self, U):
        """Given a cdf value, returns a value in original space.

        Args:
            U: `int` or `float` cdf value in [0,1]

        Returns:
            float: value in original space
        """
        self.check_fit()

        if not 0 < U < 1:
            raise ValueError('cdf value must be in [0,1]')

        return scipy.optimize.brentq(self.cumulative_distribution, -1000.0, 1000.0, args=(U))

    @random_state
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
