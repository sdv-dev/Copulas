import numpy as np
import scipy

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
        """Fit Kernel density estimation to an list of values.

        Args:
            X: 1-d `np.ndarray` or `pd.Series` or `list` datapoints to be estimated from.

        This function will fit a gaussian_kde model to a list of datapoints
        and store it as a class attribute.
        """
        if not len(X):
            raise ValueError("data cannot be empty")

        self.model = scipy.stats.gaussian_kde(X)
        self.fitted = True

    def probability_density(self, X):
        """Evaluate the estimated pdf on a point.

        Args:
            X: `float` a datapoint.
            :type X: float

        Returns:
            pdf: int or float with the value of estimated pdf
        """
        self.check_fit()
        if type(X) not in (int, float):
            raise ValueError('x must be int or float')

        return self.model.evaluate(X)[0]

    def cumulative_distribution(self, X, U=0):
        """Computes the integral of a 1-D pdf between two bounds

        Args:
            X: `float` a datapoint.
            U: `float` cdf value in [0,1], only used in get_ppf

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

    def sample(self, num_samples=1):
        """Samples new data point based on model.

        Args:
            num_samples: `int` number of points to be sampled

        Returns:
            samples: a list of datapoints sampled from the model
        """
        self.check_fit()
        return self.model.resample(num_samples)

    @classmethod
    def from_dict(cls, copula_dict):
        """Set attributes with provided values."""
        instance = cls()

        if copula_dict['fitted']:
            instance.model = scipy.stats.gaussian_kde([-1, 0, 0])

            for key in ['dataset', 'covariance', 'inv_cov']:
                copula_dict[key] = np.array(copula_dict[key])

            attributes = ['d', 'n', 'dataset', 'covariance', 'factor', 'inv_cov']
            for name in attributes:
                setattr(instance.model, name, copula_dict[name])

        instance.fitted = copula_dict['fitted']

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
