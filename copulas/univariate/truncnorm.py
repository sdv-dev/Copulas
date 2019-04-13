import scipy

from copulas import EPSILON
from copulas.univariate.base import ScipyWrapper


class TruncNorm(ScipyWrapper):
    """Wrapper around scipy.stats.truncnorm.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """

    model_class = 'truncnorm'
    unfittable_model = True
    method_map = {
        'probability_density': 'pdf',
        'cumulative_distribution': 'cdf',
        'percent_point': 'ppf',
        'sample': 'rvs'
    }

    def fit(self, X):
        """Prepare necessary params and call super().fit."""
        min_ = X.min() - EPSILON
        max_ = X.max() + EPSILON
        self.mean = X.mean()
        self.std = X.std()

        super().fit(X, min_, max_)

    def cumulative_distribution(self, X):
        return super().cumulative_distribution(X, self.mean, self.std)

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
        instance.constant_value = parameters['constant_value']

        if instance.fitted and instance.constant_value is None:
            instance.model = scipy.stats.truncnorm(parameters['a'], parameters['b'])

        return instance

    def _fit_params(self):
        """Return attributes from self.model to serialize.

        Returns:
            dict: Parameters to recreate self.model in its current fit status.
        """
        return {
            'a': self.model.a,
            'b': self.model.b
        }
