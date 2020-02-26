import numpy as np
from scipy import stats

from copulas.bivariate import select_copula
from copulas.bivariate.frank import Frank


def test_select_copula_negative_tau():
    """If tau is negative, should choose frank copula."""
    # Setup
    X = np.array([
        [0.1, 0.6],
        [0.2, 0.5],
        [0.3, 0.4],
        [0.4, 0.3]
    ])
    assert stats.kendalltau(X[:, 0], X[:, 1])[0] < 0

    # Run
    copula = select_copula(X)

    # Check
    assert isinstance(copula, Frank)
