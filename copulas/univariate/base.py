class Univariate(object):
    """ Abstract class for representing univariate distributions """

    def __init__(self):
        pass

    def fit(self, X):
        """Fits a univariate model and updates parameters.

        Arguments:
            X: `np.ndarray` of shape (n, 1) data.

        Returns:
            None
        """
        raise NotImplementedError

    def probability_density(self, X):
        """given a value, returns corresponding pdf value."""
        raise NotImplementedError

    def cumulative_density(self, X):
        """Given a value returns corresponding cdf value."""
        raise NotImplementedError

    def percent_point(self, X):
        """Given a cdf value, returns a value in original space."""
        raise NotImplementedError

    def sample(self, n_samples=1):
        """Returns new data point based on model.

        Argument:
            n_samples: `int`

        Returns:
            np.ndarray: Generated samples
        """
        raise NotImplementedError
