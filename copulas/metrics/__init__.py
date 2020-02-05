import numpy as np


def ks_statistic(copula, X):
    copula.fit(X)
    estimated_cdf = np.sort(copula.cumulative_distribution(X))
    emperical_cdf = np.linspace(0.0, 1.0, num=len(X))
    statistic = max(np.abs(estimated_cdf - emperical_cdf))
    return statistic
