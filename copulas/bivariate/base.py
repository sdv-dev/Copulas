from scipy import stats


class Bivariate(object):
    """Base class for all bivariate copulas."""

    def __init__(self, *args, **kwargs):
        """ initialize copula object """

    def fit(self, U, V):
        """ Fits a model to the data and updates the parameters """
        self.U = U
        self.V = V
        self.tau = stats.kendalltau(self.U, self.V)[0]
        self.theta = self.tau_to_theta()

    def infer(self, values):
        """ Takes in subset of values and predicts the rest """
        raise NotImplementedError

    def get_pdf(self):
        """ returns pdf of model """
        raise NotImplementedError

    def get_cdf(self):
        """ returns cdf of model """
        raise NotImplementedError

    def sample(self):
        """ returns a new data point generated from model """
        raise NotImplementedError

    def tau_to_theta(self):
        """returns theta parameter."""
        raise NotImplementedError
