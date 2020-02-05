from copulas.metrics import ks_statistic
from copulas.univariate.base import Univariate
from copulas.univariate.gaussian import GaussianUnivariate
from copulas.univariate.gaussian_kde import GaussianKDE
from copulas.univariate.kde import KDEUnivariate
from copulas.univariate.truncated_gaussian import TruncatedGaussian

__all__ = (
    'GaussianKDE',
    'GaussianUnivariate',
    'KDEUnivariate',
    'TruncatedGaussian',
    'Univariate',
)


def select_univariate(X):
    """
    This function returns an Univariate model which minimizes the KS statistic.
    """
    best_ks, best_model = float("inf"), None
    for model in [GaussianUnivariate(), GaussianKDE(), TruncatedGaussian()]:
        ks = ks_statistic(model, X)
        if ks < best_ks:
            best_ks = ks
            best_model = model
    return best_model
