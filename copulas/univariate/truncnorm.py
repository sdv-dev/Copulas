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
        self.constant_value = self._get_constant_value(X)
        if self.constant_value is None:
            min_ = X.min() - EPSILON
            max_ = X.max() + EPSILON
            mean = X.mean()
            std = X.std()

            super().fit(X, min_, max_, loc=mean, scale=std)
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
            a = parameters['a']
            b = parameters['b']
            mean = parameters['mean']
            std = parameters['std']

            if a == b == mean and std == 0:
                instance.constant_value = a

            else:
                instance.model = scipy.stats.truncnorm(a, b, loc=mean, scale=std)

        return instance

    def _fit_params(self):
        """Return attributes from self.model to serialize.

        Returns:
            dict: Parameters to recreate self.model in its current fit status.
        """
        if self.constant_value is not None:
            return {
                'a': self.constant_value,
                'b': self.constant_value,
                'mean': self.constant_value,
                'std': 0
            }

        return dict(
            a=self.model.a,
            b=self.model.b,
            **self.model.kwds
        )
