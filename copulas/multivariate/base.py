import json


class Multivariate(object):
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

    def to_dict(self):
        raise NotImplementedError

    def from_dict(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, copula):
        """Creates a new instance from a file or dict."""
        if isinstance(copula, str):
            copula = json.loads(copula)

        instance = cls()
        instance.from_dict(**copula)
        return instance

    def save(self, filename):
        """Save the internal state of a copula in the specified filename."""
        with open(filename, 'w') as f:
            f.write(json.dumps(self.to_dict()))
