from enum import Enum

import numpy as np
from scipy import stats

COMPUTE_EMPIRICAL_STEPS = 50


class CopulaTypes(Enum):
    """Available copulas  """
    CLAYTON = 0
    FRANK = 1
    GUMBEL = 2


class NotFittedError(Exception):
    pass


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
        """
        Creates a new instance of any of their subclasses.

        Args:
            copula_type: `CopulaType` or `str` to be compared against CopulaType.
        """
        self.theta = None
        self.tau = None

    def fit(self, U, V):
        """
        Fits a model to the data updating the parameters.

        Args:
            U: 1-d `np.ndarray` for first variable to train the copula.
            V: 1-d `np.ndarray` for second variable to train the copula.

        Return:
            None
        """
        self.U = U
        self.V = V
        self.tau = stats.kendalltau(self.U, self.V)[0]
        self.theta = self.get_theta()

    def infer(self, values):
        """Takes in subset of values and predicts the rest."""
        raise NotImplementedError

    def generator(self, t):
        raise NotImplementedError

    def probability_density(self, U, V):
        """Compute probability density function for given copula family.

        Args:
            U: `np.ndarray`
            V: `np.ndarray`

        Returns:
            np.array: Probability density for the input values.
        """
        raise NotImplementedError

    def cumulative_density(self, U, V):
        """Computes the cumulative distribution function for the copula, :math:`C(u, v)`

        Args:
            U: `np.ndarray`
            V: `np.ndarray`

        Returns:
            np.array: cumulative probability

        """
        raise NotImplementedError

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative density :math:`C(u|v)^-1`

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """

        raise NotImplementedError

    def partial_derivative(self, U, V, y=0):
        """Compute partial derivative :math:`C(u|v)` of cumulative density.

        Args:
            U: `np.ndarray`
            V: `np.ndarray`
            y: `float`

        Returns:

        """
        raise NotImplementedError

    def _sample(self, v, c):
        raise NotImplementedError

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

        u = self._sample(v, c)
        U = np.column_stack((u.flatten(), v))
        return U

    def get_theta(self):
        """Compute theta parameter using Kendall's tau."""
        raise NotImplementedError

    def check_fit(self):
        if not self.theta:
            raise NotFittedError("This model is not fitted.")

    @staticmethod
    def compute_tail(c, z):
        return np.divide(1.0 - 2 * np.asarray(z) + c, np.power(1.0 - np.asarray(z), 2))

    @classmethod
    def get_dependences(cls, copulas, z_left, z_right):
        left = right = []

        for copula in copulas:
            left.append(copula.copula_cumulative_density(z_left, z_left) / np.power(z_left, 2))

        for copula in copulas:
            right.append(cls.compute_tail(
                copula.copula_cumulative_density(z_right, z_right), z_right))

        return left, right

    @classmethod
    def select_copula(cls, U, V):
        """Select best copula function based on likelihood.

        Args:
            U: 1-dimensional `np.ndarray`
            V: 1-dimensional `np.ndarray`

        Returns:
            tuple: `tuple(CopulaType, float)` best fit and model param.
        """
        clayton = Bivariate(CopulaTypes.CLAYTON)
        clayton.fit(U, V)

        if clayton.tau <= 0:
            frank = Bivariate(CopulaTypes.FRANK)
            frank.fit(U, V)
            selected_theta = frank.theta
            selected_copula = frank.copula_type

            return selected_copula, selected_theta

        frank = Bivariate(CopulaTypes.FRANK)
        frank.fit(U, V)
        gumbel = Bivariate(CopulaTypes.GUMBEL)
        gumbel.fit(U, V)
        candidate_copulas = [clayton, frank, gumbel]
        theta_candidates = [clayton.theta, frank.theta, gumbel.theta]

        z_left, L, z_right, R = cls.compute_empirical(U, V)
        left_dependence, right_dependence = cls.get_dependences(candidate_copulas, z_left, z_right)

        # compute L2 distance from empirical distribution
        cost_L = [np.sum((L - l) ** 2) for l in left_dependence]
        cost_R = [np.sum((R - r) ** 2) for r in right_dependence]
        cost_LR = np.add(cost_L, cost_R)
        selected_copula = np.argmax(cost_LR)
        selected_theta = theta_candidates[selected_copula]
        return selected_copula, selected_theta

    @staticmethod
    def compute_empirical(u, v):
        """Compute empirical distribution."""
        z_left = z_right = []
        L = R = []
        N = len(u)
        base = np.linspace(0.0, 1.0, COMPUTE_EMPIRICAL_STEPS)

        for k in range(COMPUTE_EMPIRICAL_STEPS):
            left = sum(np.logical_and(u <= base[k], v <= base[k])) / N
            right = sum(np.logical_and(u >= base[k], v >= base[k])) / N

            if left > 0:
                z_left.append(base[k])
                L.append(left / base[k]**2)

            if right > 0:
                z_right.append(base[k])
                R.append(right / (1 - z_right[k])**2)

        return z_left, L, z_right, R
