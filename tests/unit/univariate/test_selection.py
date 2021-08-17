from unittest.mock import Mock, patch

import numpy as np
from scipy.stats import truncnorm

from copulas.univariate import BetaUnivariate, GaussianKDE, GaussianUnivariate, TruncatedGaussian
from copulas.univariate.selection import select_univariate


def test_select_univariate_bimodal():
    """
    Suppose the data follows a bimodal distribution. The model selector should be able to
    figure out that the GaussianKDE is best.
    """
    mask = np.random.normal(size=1000) > 0.5
    mode1 = np.random.normal(size=1000) * mask
    mode2 = np.random.normal(size=1000, loc=10) * (1.0 - mask)
    bimodal_data = mode1 + mode2

    candidates = [GaussianKDE, GaussianUnivariate, TruncatedGaussian]

    model = select_univariate(bimodal_data, candidates)

    assert isinstance(model, GaussianKDE)


def test_select_univariate_binary():
    """
    Suppose the data follows a Bernoulli distribution. The KS statistic should be larger
    for a TruncatedGaussian model than a GaussianKDE model which can somewhat capture a
    Bernoulli distribution as it resembles a bimodal distribution.
    """
    candidates = [GaussianKDE(), TruncatedGaussian()]

    binary_data = np.random.randint(0, 2, size=10000)

    model = select_univariate(binary_data, candidates)

    assert isinstance(model, GaussianKDE)


def test_select_univariate_truncated():
    """
    Suppose the data follows a truncated normal distribution. The KS statistic should be
    larger for a Gaussian model than a TruncatedGaussian model (since the fit is worse).
    """
    a, b, loc, scale = -1.0, 0.5, 0.0, 1.0
    truncated_data = truncnorm.rvs(a, b, loc=loc, scale=scale, size=10000)

    candidates = [GaussianUnivariate(), TruncatedGaussian()]

    model = select_univariate(truncated_data, candidates)

    assert isinstance(model, TruncatedGaussian)


@patch('copulas.univariate.selection.get_instance')
def test_select_univariate_failures(get_instance_mock):
    """Make sure that failed candidates are discarded.

    Whenever there is a crash on a candidate, it should be discarded from the list.

    An example of a possible failure is the one described in issue
    https://github.com/sdv-dev/Copulas/issues/264

    For this test, GaussianUnivariate is mocked to raise an Exception during fit.
    Because of this, a BetaUnivariate will be selected instead of Gaussian,
    even though the data follows a Gaussian distribution.

    Setup:
        - Mock GaussianUnivariate to raise an Exception during fit
    Input:
        - Numpy array following a normal distribution
        - Candidates list that includes Beta and Gaussian
    Output:
        - Beta instance.
    """
    gaussian_mock = Mock()
    gaussian_mock.fit.side_effect = Exception()
    get_instance_mock.side_effect = [
        gaussian_mock,
        BetaUnivariate(),
        BetaUnivariate()
    ]
    normal_data = np.random.normal(size=1000)

    candidates = [GaussianUnivariate, BetaUnivariate]

    model = select_univariate(normal_data, candidates)

    assert isinstance(model, BetaUnivariate)
