import numpy as np


def ks_statistic(distribution, X):
    distribution.fit(X)
    estimated_cdf = np.sort(distribution.cdf(X))
    emperical_cdf = np.linspace(0.0, 1.0, num=len(X))
    statistic = max(np.abs(estimated_cdf - emperical_cdf))
    return statistic


def select_univariate(X, parametric=None):
    from copulas.univariate import (
        BetaUnivariate, GammaUnivariate, GaussianUnivariate, GaussianKDE, TruncatedGaussian)
    models = [
        BetaUnivariate(),
        GammaUnivariate(),
        GaussianUnivariate(),
        TruncatedGaussian(),
        GaussianKDE()]

    best_ks = np.inf
    best_model = None
    for model in models:
        if not (parametric is None or model.parametric == parametric):
            continue
        ks = ks_statistic(model, X)
        if ks < best_ks:
            best_ks = ks
            best_model = model

    return best_model
