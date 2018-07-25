import logging

import numpy as np

from copulas.univariate.UnivariateDistrib import UnivariateDistrib
from scipy.stats import gaussian_kde
import scipy.optimize as optimize


LOGGER = logging.getLogger(__name__)


class KDEUnivariate(UnivariateDistrib):
    """ Use Kernel Density Estimation to infer univariate distribution """

    def __init__(self):
        super(KDEUnivariate, self).__init__()
        self.column = None
        self.param = None

    def fit(self, column):
        self.column = column
        model = gaussian_kde(column)
        self.param = model

    def get_pdf(self, x):
        return self.param.evaluate(x)

    def get_cdf(self, x, u=0):
        low_bounds = -10000
        return self.param.integrate_box_1d(low_bounds, x) - u

    def get_ppf(self, u):
        """ given a cdf value, returns a value in original space """
        return optimize.brentq(self.get_cdf, -1000.0, 1000.0, args=(u))

    def sample(self, num_samples=1):
        """ returns new data point based on model """
        return self.param.resample(num_samples)
