import numpy as np

from copulas.univariate.base import Univariate
from copulas.univariate.beta import BetaUnivariate
from copulas.univariate.gamma import GammaUnivariate
from copulas.univariate.gaussian import GaussianUnivariate
from copulas.univariate.gaussian_kde import GaussianKDE
from copulas.univariate.truncated_gaussian import TruncatedGaussian

__all__ = (
    'select_univariate',
    'BetaUnivariate',
    'GammaUnivariate',
    'GaussianKDE',
    'GaussianUnivariate',
    'TruncatedGaussian',
    'Univariate',
)


def ks_statistic(distribution, X):
    distribution.fit(X)
    estimated_cdf = np.sort(distribution.cdf(X))
    emperical_cdf = np.linspace(0.0, 1.0, num=len(X))
    statistic = max(np.abs(estimated_cdf - emperical_cdf))
    return statistic


def select_univariate(X):
    """
    This function returns an Univariate model which minimizes the KS statistic.
    """
    best_ks, best_model = float("inf"), None
    for model in [BetaUnivariate(), GammaUnivariate(), GaussianUnivariate(),
                  GaussianKDE(), TruncatedGaussian()]:
        ks = ks_statistic(model, X)
        if ks < best_ks:
            best_ks = ks
            best_model = model
    return best_model
