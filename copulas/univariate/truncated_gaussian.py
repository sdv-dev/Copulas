import scipy

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

    def __init__(self, min=None, max=None, epsilon=0):
        super(TruncatedGaussian, self).__init__()
        self._min = min
        self._max = max
        self.epsilon = epsilon

    def fit(self, X):
        """Prepare necessary params and call super().fit."""
        if self._min is None:
            self._min = X.min() - self.epsilon

        if self._max is None:
            self._max = X.max() + self.epsilon

        self._std = X.std()
        self._mean = X.mean()

        super().fit(X, self._min, self._max, loc=self._std, scale=self._mean)
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
            min_ = parameters['min']
            max_ = parameters['max']
            std = parameters['std']
            mean = parameters['mean']
            epsilon = parameters['epsilon']

            if min_ == max_:
                instance.constant_value = min_

            else:
                instance._min = min_
                instance._max = max_
                instance._std = std
                instance._mean = mean
                instance.epsilon = epsilon
                instance.model = scipy.stats.truncnorm(min_, max_, loc=std, scale=mean)

            cls._replace_methods(instance)

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
            }

        return {
            'min': self._min,
            'max': self._max,
            'std': self._std,
            'mean': self._mean,
            'epsilon': self.epsilon
        }
