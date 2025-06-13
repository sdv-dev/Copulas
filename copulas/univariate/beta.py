"""BetaUnivariate module."""

import numpy as np
from scipy.stats import beta

from copulas.univariate.base import BoundedType, ParametricType, ScipyModel
from copulas.utils import EPSILON


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
        min_x = np.min(X)
        max_x = np.max(X)
        a, b, loc, scale = beta.fit(X, loc=min_x, scale=max_x - min_x)

        if loc > max_x or scale + loc < min_x:
            raise ValueError(
                'Converged parameters for beta distribution are '
                'outside the min/max range of the data.'
            )

        if scale < EPSILON:
            raise ValueError('Converged parameters for beta distribution have a near-zero range.')

        self._params = {'loc': loc, 'scale': scale, 'a': a, 'b': b}

    def _is_constant(self):
        return self._params['scale'] == 0

    def _extract_constant(self):
        return self._params['loc']
