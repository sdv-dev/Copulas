import numpy as np

from copulas import get_instance


def ks_statistic(model, X):
    instance = get_instance(model)
    instance.fit(X)
    estimated_cdf = np.sort(instance.cdf(X))
    emperical_cdf = np.linspace(0.0, 1.0, num=len(X))
    return max(np.abs(estimated_cdf - emperical_cdf))


def select_univariate(X, candidates):
    best_ks = np.inf
    best_model = None
    for model in candidates:
        ks = ks_statistic(model, X)
        if ks < best_ks:
            best_ks = ks
            best_model = model

    return get_instance(best_model)
