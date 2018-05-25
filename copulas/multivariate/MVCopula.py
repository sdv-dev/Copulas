class MVCopula:
    """ Abstract class for a multi-variate copula object """

    def __init__(self):
        """ initialize copula object """

    def fit(self, data):
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

    def sample(self, num_rows=1):
        """ returns a new data point generated from model """
        raise NotImplementedError
