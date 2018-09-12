class Univariate(object):
    """ Abstract class for representing univariate distributions """

    def __init__(self):
        pass

    def fit(self, X):
        """Fits the model.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            None
        """
        raise NotImplementedError

    def probability_density(self, X):
        """Computes probability density.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray
        """
        raise NotImplementedError

    def pdf(self, X):
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Computes cumulative density.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray: Cumulative density for X.
        """
        raise NotImplementedError

    def cdf(self, X):
        return self.cumulative_distribution(X)

    def percent_point(self, U):
        """Given a cumulated density, returns a value in original space.

        Arguments:
            U: `np.ndarray` of shape (n, 1) and values in [0,1]

        Returns:
            `np.ndarray`: Estimated values in original space.
        """
        raise NotImplementedError

    def ppf(self, U):
        return self.percent_point(U)

    def sample(self, n_samples=1):
        """Returns new data point based on model.

        Argument:
            n_samples: `int`

        Returns:
            np.ndarray: Generated samples
        """
        raise NotImplementedError

    def to_dict(self):
        """Returns parameters to replicate the distribution."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, param_dict):
        """Create new instance from dictionary."""
        raise NotImplementedError
