from copulas.univariate.base import ScipyWrapper


class TruncNorm(ScipyWrapper):
    """Wrapper around scipy.stats.truncnorm.
    
    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """

    model_class = 'truncnorm'
    method_map = {
        'probability_density': 'pdf',
        'cumulative_distribution': 'cdf',
        'percent_point': 'ppf',
        'sample': 'rvs'
    }

    @classmethod
    def from_dict(cls, parameters):
        """Set attributes with provided values.

        Args:
            parameters(dict): Dictionary containing instance parameters.

        Returns:
            Truncnorm: Instance populated with given parameters.
        """

    def _fit_params(self):
        """Return attributes from self.model to serialize.

        Returns:
            dict: Parameters to recreate self.model in its current fit status.
        """