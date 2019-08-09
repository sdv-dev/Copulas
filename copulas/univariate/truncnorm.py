import scipy

from copulas import EPSILON
from copulas.univariate.base import ScipyWrapper


class TruncNorm(ScipyWrapper):
    """Wrapper around scipy.stats.truncnorm.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """

    model_class = 'truncnorm'
    unfittable_model = True
    probability_density = 'pdf'
    cumulative_distribution = 'cdf'
    percent_point = 'ppf'
    sample = 'rvs'

    def fit(self, X):
        """Prepare necessary params and call super().fit."""
        min_ = X.min() - EPSILON
        max_ = X.max() + EPSILON

        super().fit(X, min_, max_)

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
            a = parameters['a']
            b = parameters['b']

            if a > b:
                instance.constant_value = a

            else:
                instance.model = scipy.stats.truncnorm(a, b)

        return instance

    def _fit_params(self):
        """Return attributes from self.model to serialize.

        Returns:
            dict: Parameters to recreate self.model in its current fit status.
        """
        if self.constant_value is not None:
            return {
                'a': self.constant_value,
                'b': self.constant_value - 1
            }

        return {
            'a': self.model.a,
            'b': self.model.b
        }
