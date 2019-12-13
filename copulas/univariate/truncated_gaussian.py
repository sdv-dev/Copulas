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
        self.min = min
        self.max = max
        self.epsilon = epsilon

    def fit(self, X):
        """Prepare necessary params and call super().fit."""
        if self.min is None:
            self.min = X.min() - self.epsilon

        if self.max is None:
            self.max = X.max() + self.epsilon

        self.std = X.std()
        self.mean = X.mean()

        super().fit(X, self.min, self.max, loc=self.std, scale=self.mean)
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
                instance.min = min_
                instance.max = max_
                instance.std = std
                instance.mean = mean
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
            'min': self.min,
            'max': self.max,
            'std': self.std,
            'mean': self.mean,
            'epsilon': self.epsilon
        }
