import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.stats import truncnorm

from copulas import EPSILON, check_valid_values
from copulas.univariate.base import ScipyWrapper


class TruncatedGaussian(ScipyWrapper):
    """Wrapper around scipy.stats.truncnorm.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """

    model_class = 'truncnorm'
    unfittable_model = True
    probability_density = 'pdf'
    cumulative_distribution = 'cdf'
    percent_point = 'ppf'
    sample = 'rvs'

    fitted = False
    constant_value = None
    mean = None
    std = None

    def __init__(self, min=None, max=None, random_seed=None):
        self.random_seed = random_seed
        self.min = min
        self.max = max

    def _get_model(self):
        self.a = (self.min - self.mean) / self.std
        self.b = (self.max - self.mean) / self.std
        return truncnorm(self.a, self.b, loc=self.mean, scale=self.std)

    def _fit_truncnorm(self, X):
        """Fit the truncnorm parameters to the data.

        The optimization is done using a constraint to force a loc value
        that strictly stays between the minimum and maximum values. This is done
        to avoid falling into degenerated situations where the mean is far away
        from the range of valid values, which provokes a ValueError inside
        scipy.truncnorm.
        """
        if self.min is None:
            self.min = X.min() - EPSILON

        if self.max is None:
            self.max = X.max() + EPSILON

        def nnlf(params):
            loc, scale = params
            a = (self.min - loc) / scale
            b = (self.max - loc) / scale
            return truncnorm.nnlf((a, b, loc, scale), X)

        def constraint(params):
            loc, scale = params
            a = (self.min - loc) / scale
            b = (self.max - loc) / scale
            return np.array([a < 0, 0 < b]).astype(int)

        initial_params = X.mean(), X.std()
        optimal = fmin_slsqp(nnlf, initial_params, f_eqcons=constraint, iprint=False)

        self.mean, self.std = optimal
        self.model = self._get_model()

    @check_valid_values
    def fit(self, X):
        """Fit scipy model to an array of values.

        Args:
            X(`np.ndarray` or `pd.DataFrame`):  Datapoints to be estimated from. Must be 1-d

        Returns:
            None
        """

        self.constant_value = self._get_constant_value(X)

        if self.constant_value is None:
            self._fit_truncnorm(X)
            self._replace_methods()
        else:
            self._replace_constant_methods()

        self.fitted = True

    @classmethod
    def from_dict(cls, parameters):
        """Set attributes with provided values.

        Args:
            parameters(dict): Dictionary containing instance parameters.

        Returns:
            Truncnorm: Instance populated with given parameters.
        """
        instance = cls()
        instance.fitted = parameters['fitted']

        if instance.fitted:
            instance.min = parameters['min']
            instance.max = parameters['max']
            instance.std = parameters['std']
            instance.mean = parameters['mean']

            if instance.min == instance.max:
                instance.constant_value = instance.min

            else:
                instance.model = instance._get_model()
                instance._replace_methods()

        return instance

    def _fit_params(self):
        """Return attributes from self.model to serialize.

        Returns:
            dict: Parameters to recreate self.model in its current fit status.
        """
        if self.constant_value is not None:
            return {
                'min': self.constant_value,
                'max': self.constant_value,
                'std': 0,
                'mean': self.constant_value,
            }

        return {
            'min': self.min,
            'max': self.max,
            'std': self.std,
            'mean': self.mean,
        }
