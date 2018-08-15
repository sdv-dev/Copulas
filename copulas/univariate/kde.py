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

    def probability_density(self, x):
        """Evaluate the estimated pdf on a point.

        Args:
            :param x: a datapoint.
            :type x: int or float

        Returns:
            pdf: int or float with the value of estimated pdf
        """
        if type(x) not in (int, float):
            raise ValueError('x must be int or float')

        return self.model.evaluate(x)[0]

    def cumulative_density(self, x, u=0):
        """Computes the integral of a 1-D pdf between two bounds

        Args:
            :param x: a datapoint.
            :param u: cdf value in [0,1], only used in get_ppf
            :type x: int or float
            :type u: int or float

        Returns:
            cdf: int or float with the value of estimated cdf.
        """
        low_bounds = -10000
        return self.model.integrate_box_1d(low_bounds, x) - u

    def percent_point(self, u):
        """Given a cdf value, returns a value in original space.

        Args:
            u: `int` or `float` cdf value in [0,1]

        Returns:
            float: value in original space
        """
        if u <= 0 or u >= 1:
            raise ValueError('cdf value must be in [0,1]')

        return scipy.optimize.brentq(self.cumulative_density, -1000.0, 1000.0, args=(u))

    def sample(self, n_samples=1):
        """Samples new data point based on model.

        Args:
            n_samples: `int` number of points to be sampled

        Returns:
            samples: a list of datapoints sampled from the model
        """
        return self.model.resample(n_samples)
