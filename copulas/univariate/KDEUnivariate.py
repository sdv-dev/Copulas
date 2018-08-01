import scipy

from copulas.univariate.UnivariateDistrib import UnivariateDistrib


class KDEUnivariate(UnivariateDistrib):
    """ A wrapper for gaussian Kernel density estimation implemented
    in scipy.stats toolbox. gaussian_kde is slower than statsmodels
    but allows more flexibility.
    """

    def __init__(self):
        super(KDEUnivariate, self).__init__()
        self.model = None

    def fit(self, column):
        """Fit Kernel density estimation to an list of values.
        Args:
            :param column: list of datapoints to be estimated from.
            :type column: 1-D array
        This function will fit a gaussian_kde model to a list of datapoints
        and store it as a class attribute.
        """
        if column is None:
            raise ValueError("data cannot be empty")
        self.model = scipy.stats.gaussian_kde(column)

    def get_pdf(self, x):
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

    def get_cdf(self, x, u=0):
        """Computes the integral of a 1D pdf between two bounds
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

    def get_ppf(self, u):
        """ Given a cdf value, returns a value in original space
        Args:
            :param u: cdf value in [0,1]
            :type u: int or float
        Returns:
            x: int or float with the value in original space
        """
        return scipy.optimize.brentq(self.get_cdf, -1000.0, 1000.0, args=(u))

    def sample(self, num_samples=1):
        """ Samples new data point based on model
        Args:
            :param num_samples: number of points to be sampled
            :type num_samples: int
        Returns:
            samples: a list of datapoints sampled from the model
        """
        return self.model.resample(num_samples)
