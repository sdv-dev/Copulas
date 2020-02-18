from scipy.stats import beta

from copulas import check_valid_values
from copulas.univariate.base import ScipyWrapper


class BetaUnivariate(ScipyWrapper):
    """Wrapper around scipy.stats.beta.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
    """

    model_class = 'beta'
    unfittable_model = True
    probability_density = 'pdf'
    cumulative_distribution = 'cdf'
    percent_point = 'ppf'
    sample = 'rvs'

    fitted = False
    parametric = True
    constant_value = None

    def __init__(self):
        self.a = None
        self.b = None
        self.loc = None
        self.scale = None

    def _get_model(self):
        return beta(self.a, self.b, loc=self.loc, scale=self.scale)

    def _fit_beta(self, X):
        """Fit the beta parameters to the data.
        """
        self.a, self.b, self.loc, self.scale = beta.fit(X)
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
            self._fit_beta(X)
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
            instance.b = parameters['b']
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
                'a': 0.0,
                'b': 0.0,
                'loc': self.constant_value,
                'scale': 0.0,
            }

        return {
            'a': self.a,
            'b': self.b,
            'loc': self.loc,
            'scale': self.scale,
        }
