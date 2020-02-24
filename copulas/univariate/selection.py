import numpy as np

from scipy.stats import kstest
from copulas import get_instance

def select_univariate(X, candidates):
    best_ks = np.inf
    best_model = None
    for model in candidates:
        instance = get_instance(model)
        instance.fit(X)
        ks, _ = kstest(X, instance.cdf)
        if ks < best_ks:
            best_ks = ks
            best_model = model

    return get_instance(best_model)
