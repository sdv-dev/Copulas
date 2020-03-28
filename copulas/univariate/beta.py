import numpy as np
from scipy.stats import beta

from copulas.univariate.base import BoundedType, ParametricType, ScipyModel


class BetaUnivariate(ScipyModel):
    """Wrapper around scipy.stats.beta.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
    """

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.BOUNDED
    MODEL_CLASS = beta

    def _fit_constant(self, X):
        self._params = {
            'a': 1.0,
            'b': 1.0,
            'loc': np.unique(X)[0],
            'scale': 0.0,
        }

    def _fit(self, X):
        loc = np.min(X)
        scale = np.max(X) - loc
        a, b = beta.fit(X, loc=loc, scale=scale)[0:2]
        self._params = {
            'loc': loc,
            'scale': scale,
            'a': a,
            'b': b
        }

    def _is_constant(self):
        return self._params['scale'] == 0
