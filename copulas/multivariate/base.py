import json

from copulas import NotFittedError, import_object


class Multivariate(object):
    """Abstract class for a multi-variate copula object."""

    def __init__(self):
        """Initialize copula object."""
        self.fitted = False

    def fit(self, X):
        """Fit a model to the data and update the parameters."""
        raise NotImplementedError

    def infer(self, values):
        """Predict data from a subset of values."""
        raise NotImplementedError

    def probability_density(self, X):
        """Return probability density of model."""
        raise NotImplementedError

    def pdf(self, X):
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Return cumulative density of model."""
        raise NotImplementedError

    def cdf(self, X):
        return self.cumulative_distribution(X)

    def sample(self, num_rows=1):
        """Return a new data point generated from model."""
        raise NotImplementedError

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this object.

        Args:
            self:

        Returns:
            dict: Parameters of the copula.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, copula_dict):
        """Create a new instance from dictionary.

        Args:
            copula_dict: `dict` with the parameters to replicate the copula.
            Like the output of `Multivariate.to_dict`

        Returns:
            Multivariate: Instance of the copula defined on the parameters.
        """

        copula_class = import_object(copula_dict['type'])
        return copula_class.from_dict(copula_dict)

    @classmethod
    def load(cls, copula_path):
        """Create a new instance from a file.

        Args:
            copula_path: `str` file with the serialized copula.

        Returns:
            Bivariate: Instance with the parameters stored in the file.
        """
        with open(copula_path) as f:
            copula_dict = json.load(f)

        return cls.from_dict(copula_dict)

    def save(self, filename):
        """Save the internal state of a copula in the specified filename.

        Args:
            filename: `str` path to save.

        Returns:
            None
        """
        content = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(content, f)

    def check_fit(self):
        """Assert that the object is fit

        Raises a `NotFittedError` if the model is  not fitted.
        """
        if not self.fitted:
            raise NotFittedError("This model is not fitted.")
