class BVCopula:
    """ Abstract class for a bivariate copula object """

    def __init__(self, data):
        """ initialize copula object """
        self.data = data

    def fit(self):
        """ Fits a model to the data and updates the parameters """
        raise NotImplementedError

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
