import json
from enum import Enum

import numpy as np
from scipy import stats

COMPUTE_EMPIRICAL_STEPS = 50


class CopulaTypes(Enum):
    """Available copulas  """
    CLAYTON = 0
    FRANK = 1
    GUMBEL = 2


class Bivariate(object):
    """Base class for all bivariate copulas."""

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

        >>> Bivariate(CopulaTypes.FRANK).__class__
        copulas.bivariate.frank.Frank

        >>> Bivariate('frank').__class__
        copulas.bivariate.frank.Frank
        """

    def fit(self, U, V):
        """
        Fits a model to the data updating the parameters.

        Args:
            U: 1-d `np.ndarray` for first variable to train the copula.
            V: 1-d `np.ndarray` for second variable to train the copula.

        Return:
            `None`
        """
        self.U = U
        self.V = V
        self.tau = stats.kendalltau(self.U, self.V)[0]
        self.theta = self.tau_to_theta()

    def to_dict(self):
        return {
            'copula_type': self.copula_type.name,
            'theta': self.theta,
            'tau': self.tau
        }

    @classmethod
    def from_dict(cls, **kwargs):
        instance = cls(kwargs['copula_type'])
        instance.theta = kwargs['theta']
        instance.tau = kwargs.get('tau')
        return instance

    def infer(self, values):
        """Takes in subset of values and predicts the rest."""
        raise NotImplementedError

    def get_pdf(self):
        """Returns pdf of model."""
        raise NotImplementedError

    def get_cdf(self):
        """Returns cdf of model."""
        raise NotImplementedError

    def _sample(self, v, c):
        raise NotImplementedError

    def sample(self, n_samples):
        """Generate specified `n_samples` of new data from model. `v~U[0,1],v~C^-1(u|v)`

        Args:
            n_samples: `int`, amount of samples to create.

        Returns:
            np.ndarray with generated samples.
        """
        if self.tau > 1 or self.tau < -1:
            raise ValueError("The range for correlation measure is [-1,1].")

        v = np.random.uniform(0, 1, n_samples)
        c = np.random.uniform(0, 1, n_samples)

        u = self._sample(v, c)
        U = np.column_stack((u.flatten(), v))
        return U

    def tau_to_theta(self):
        """Compute theta parameter."""
        raise NotImplementedError

    @staticmethod
    def g(c, z):
        return np.divide(1.0 - 2 * np.asarray(z) + c, np.power(1.0 - np.asarray(z), 2))

    @classmethod
    def get_dependences(cls, copulas, z_left, z_right):
        left = right = []

        for copula in copulas:
            left.append(copula.get_cdf()(z_left, z_left) / np.power(z_left, 2))

        for copula in copulas:
            right.append(cls.g(copula.get_cdf()(z_right, z_right), z_right))

        return left, right

    @classmethod
    def select_copula(cls, U, V):
        """Select best copula function based on likelihood.

        Args:
            U: 1-dimensional `np.ndarray`
            V: 1-dimensional `np.ndarray`

        Returns:
            selected_copula: `CopulaType` that best  Best fit.
            selected_theta: `float` computed for input data.
        """
        clayton = Bivariate(CopulaTypes.CLAYTON)
        clayton.fit(U, V)

        if clayton.tau <= 0:
            frank = Bivariate(CopulaTypes.FRANK)
            frank.fit(U, V)
            selected_theta = frank.theta
            selected_copula = CopulaTypes.FRANK

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
        return CopulaTypes(selected_copula), selected_theta

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

    def save(self, filename):
        """Save the internal state of a copula in the specified filename."""
        content = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(content, f)

    @classmethod
    def load(cls, copula_path):
        """Creates a new instance from a file or dict."""
        with open(copula_path) as f:
            copula_dict = json.load(f)

        return cls.from_dict(**copula_dict)
