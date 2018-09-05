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

    @classmethod
    def from_dict(cls, copula_dict):
        raise NotImplementedError

    @classmethod
    def load(cls, copula_path):
        """Create a new instance from a file."""
        with open(copula_path) as f:
            copula_dict = json.load(f)

        return cls.from_dict(copula_dict)

    def save(self, filename):
        """Save the internal state of a copula in the specified filename."""
        content = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(content, f)
