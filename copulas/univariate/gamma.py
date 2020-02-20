from scipy.stats import gamma

from copulas import check_valid_values, store_args
from copulas.univariate.base import BoundedType, ParametricType, ScipyWrapper


class GammaUnivariate(ScipyWrapper):
    """Wrapper around scipy.stats.gamma.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    """

    model_class = 'gamma'
    unfittable_model = True
    probability_density = 'pdf'
    cumulative_distribution = 'cdf'
    percent_point = 'ppf'
    sample = 'rvs'

    fitted = False
    constant_value = None

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.SEMI_BOUNDED

    @store_args
    def __init__(self):
        self.a = None
        self.loc = None
        self.scale = None

    def _get_model(self):
        return gamma(self.a, loc=self.loc, scale=self.scale)

    def _fit_expon(self, X):
        """Fit the gamma parameters to the data."""
        self.a, self.loc, self.scale = gamma.fit(X)
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
            self._fit_expon(X)
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
            instance.a = parameters['a']
            instance.loc = parameters['loc']
            instance.scale = parameters['scale']

            if instance.scale == 0.0:
                instance.constant_value = instance.loc
                instance._replace_constant_methods()

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
                'a': 0,
                'loc': self.constant_value,
                'scale': 0.0,
            }

        return {
            'a': self.a,
            'loc': self.loc,
            'scale': self.scale,
        }
