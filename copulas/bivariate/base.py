import json
from enum import Enum

import numpy as np
from scipy import stats

from copulas import EPSILON, NotFittedError

COMPUTE_EMPIRICAL_STEPS = 50


class CopulaTypes(Enum):
    """Available copulas  """
    CLAYTON = 0
    FRANK = 1
    GUMBEL = 2


class Bivariate(object):
    """Base class for all bivariate copulas.

    This class allows to instantiate all its subclasses and serves as a unique entry point for
    all the bivariate copulas classes.

    >>> Bivariate(CopulaTypes.FRANK).__class__
    copulas.bivariate.frank.Frank

    >>> Bivariate('frank').__class__
    copulas.bivariate.frank.Frank
    """

    copula_type = None
    _subclasses = []
    theta_interval = []
    invalid_thetas = []

    @classmethod
    def _get_subclasses(cls):
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass._get_subclasses())

        return subclasses

    @classmethod
    def subclasses(cls):
        if not cls._subclasses:
            cls._subclasses = cls._get_subclasses()

        return cls._subclasses

    def __new__(cls, copula_type=None):
        if not isinstance(copula_type, CopulaTypes):
            if (isinstance(copula_type, str) and copula_type.upper() in CopulaTypes.__members__):
                copula_type = CopulaTypes[copula_type.upper()]
            else:
                raise ValueError('Invalid copula type {}'.format(copula_type))

        for subclass in cls.subclasses():
            if subclass.copula_type is copula_type:
                return super(Bivariate, cls).__new__(subclass)

    def __init__(self, copula_type=None):
        """Create a new instance of any of their subclasses.

        Args:
            copula_type: `CopulaType` or `str` to be compared against CopulaType.
        """
        self.theta = None
        self.tau = None

    def fit(self, X):
        """Fit a model to the data updating the parameters.

        Args:
            X: `np.ndarray` of shape (,2).

        Return:
            None
        """
        U, V = self.split_matrix(X)
        self.tau = stats.kendalltau(U, V)[0]
        self.theta = self.compute_theta()
        self.check_theta()

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this object.

        Args:
            self:

        Returns:
            dict: Parameters of the copula.
        """
        return {
            'copula_type': self.copula_type.name,
            'theta': self.theta,
            'tau': self.tau
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Create a new instance from the given parameters.

        Args:
            copula_dict: `dict` with the parameters to replicate the copula.
            Like the output of `Bivariate.to_dict`

        Returns:
            Bivariate: Instance of the copula defined on the parameters.
        """
        instance = cls(copula_dict['copula_type'])
        instance.theta = copula_dict['theta']
        instance.tau = copula_dict['tau']
        return instance

    def infer(self, X):
        """Take in subset of values and predicts the rest."""
        raise NotImplementedError

    def generator(self, t):
        raise NotImplementedError

    def probability_density(self, X):
        """Compute probability density function for given copula family.

        Args:
            X: `np.ndarray`

        Returns:
            np.array: Probability density for the input values.
        """
        raise NotImplementedError

    def pdf(self, X):
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Computes the cumulative distribution function for the copula, :math:`C(u, v)`.

        Args:
            X: `np.ndarray`

        Returns:
            np.array: cumulative probability
        """
        raise NotImplementedError

    def cdf(self, X):
        return self.cumulative_distribution(X)

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative density :math:`C(u|v)^-1`.

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            V: `np.ndarray` given value of V.

        Returns:
            np.ndarray: Percentiles for the given values.
        """

        raise NotImplementedError

    def ppf(self, y, V):
        return self.percent_point(y, V)

    def partial_derivative(self, X, y=0):
        """Compute partial derivative :math:`C(u|v)` of cumulative density.

        Args:
            X: `np.ndarray`
            y: `float`

        Returns:
            np.ndarray
        """
        raise NotImplementedError

    def partial_derivative_scalar(self, U, V, y=0):
        """Compute partial derivative :math:`C(u|v)` of cumulative density of single values."""

        self.check_fit()

        X = np.column_stack((U, V))
        return self.partial_derivative(X, y)

    def sample(self, n_samples):
        """Generate specified `n_samples` of new data from model. `v~U[0,1],v~C^-1(u|v)`

        Args:
            n_samples: `int`, amount of samples to create.

        Returns:
            np.ndarray: Array of length `n_samples` with generated data from the model.
        """
        if self.tau > 1 or self.tau < -1:
            raise ValueError("The range for correlation measure is [-1,1].")

        v = np.random.uniform(0, 1, n_samples)
        c = np.random.uniform(0, 1, n_samples)

        u = self.percent_point(c, v)
        return np.column_stack((u, v))

    def compute_theta(self):
        """Compute theta parameter using Kendall's tau."""
        raise NotImplementedError

    def check_theta(self):
        """Validate the computed theta against the copula specification.

        This method is used to assert the computed theta is in the valid range for the copula."""
        lower, upper = self.theta_interval
        if (not lower <= self.theta <= upper) or (self.theta in self.invalid_thetas):
            message = 'The computed theta value {} is out of limits for the given {} copula.'
            raise ValueError(message.format(self.theta, self.copula_type.name))

    def check_fit(self):
        """Assert that the model is fit and the computed `theta` is valid.

        Raises a `NotFittedError` if the model is  not fitted.
        Raises a `ValueError` if the computed theta is invalid."""
        if not self.theta:
            raise NotFittedError("This model is not fitted.")

        self.check_theta()

    @staticmethod
    def split_matrix(X):
        if len(X):
            return X[:, 0], X[:, 1]

        return np.array([]), np.array([])

    @classmethod
    def compute_empirical(cls, X):
        """Compute empirical distribution."""
        z_left = []
        z_right = []
        L = []
        R = []

        U, V = cls.split_matrix(X)
        N = len(U)
        base = np.linspace(EPSILON, 1.0 - EPSILON, COMPUTE_EMPIRICAL_STEPS)
        # See https://github.com/DAI-Lab/Copulas/issues/45

        for k in range(COMPUTE_EMPIRICAL_STEPS):
            left = sum(np.logical_and(U <= base[k], V <= base[k])) / N
            right = sum(np.logical_and(U >= base[k], V >= base[k])) / N

            if left > 0:

                z_left.append(base[k])
                L.append(left / base[k] ** 2)

            if right > 0:
                z_right.append(base[k])
                R.append(right / (1 - z_right[k]) ** 2)

        return z_left, L, z_right, R

    @staticmethod
    def compute_tail(c, z):
        return np.divide(1.0 - 2 * np.asarray(z) + c, np.power(1.0 - np.asarray(z), 2))

    @classmethod
    def get_dependencies(cls, copulas, z_left, z_right):
        left = []
        right = []

        X_left = np.column_stack((z_left, z_left))
        for copula in copulas:
            left.append(copula.cumulative_distribution(X_left) / np.power(z_left, 2))

        X_right = np.column_stack((z_right, z_right))
        for copula in copulas:
            right.append(cls.compute_tail(copula.cumulative_distribution(X_right), z_right))

        return left, right

    @classmethod
    def select_copula(cls, X):
        """Select best copula function based on likelihood.

        Args:
            X: 2-dimensional `np.ndarray`

        Returns:
            tuple: `tuple(CopulaType, float)` best fit and model param.
        """
        frank = Bivariate(CopulaTypes.FRANK)
        frank.fit(X)

        if frank.tau <= 0:
            selected_theta = frank.theta
            selected_copula = CopulaTypes.FRANK
            return selected_copula, selected_theta

        copula_candidates = [frank]
        theta_candidates = [frank.theta]

        try:
            clayton = Bivariate(CopulaTypes.CLAYTON)
            clayton.fit(X)
            copula_candidates.append(clayton)
            theta_candidates.append(clayton.theta)
        except ValueError:
            # Invalid theta, copula ignored
            pass

        try:
            gumbel = Bivariate(CopulaTypes.GUMBEL)
            gumbel.fit(X)
            copula_candidates.append(gumbel)
            theta_candidates.append(gumbel.theta)
        except ValueError:
            # Invalid theta, copula ignored
            pass

        z_left, L, z_right, R = cls.compute_empirical(X)
        left_dependence, right_dependence = cls.get_dependencies(
            copula_candidates, z_left, z_right)

        # compute L2 distance from empirical distribution
        cost_L = [np.sum((L - l) ** 2) for l in left_dependence]
        cost_R = [np.sum((R - r) ** 2) for r in right_dependence]
        cost_LR = np.add(cost_L, cost_R)

        selected_copula = np.argmax(cost_LR)
        selected_theta = theta_candidates[selected_copula]
        return CopulaTypes(selected_copula), selected_theta

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
