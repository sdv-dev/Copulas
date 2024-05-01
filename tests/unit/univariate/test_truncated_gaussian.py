import warnings
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from copulas.univariate.truncated_gaussian import TruncatedGaussian
from scipy.stats import truncnorm


class TestTruncatedGaussian(TestCase):
    def test__fit_constant(self):
        distribution = TruncatedGaussian()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {'a': 1, 'b': 1, 'loc': 1, 'scale': 0}

    def test__fit(self):
        distribution = TruncatedGaussian()

        data = truncnorm.rvs(size=10000, a=0, b=3, loc=3, scale=1)
        distribution._fit(data)

        expected = {'loc': 3, 'scale': 1, 'a': 0, 'b': 3}
        for key, value in distribution._params.items():
            np.testing.assert_allclose(value, expected[key], atol=0.3)

    @patch('copulas.univariate.truncated_gaussian.fmin_slsqp')
    def test__fit_silences_warnings(self, mocked_wrapper):
        """Test the ``_fit`` method does not emit RuntimeWarnings."""

        # Setup
        def mock_fmin_sqlsqp(*args, **kwargs):
            warnings.warn(message='Runtime Warning occured!', category=RuntimeWarning)
            return 0, 1

        mocked_wrapper.side_effect = mock_fmin_sqlsqp
        distribution = TruncatedGaussian()

        data = truncnorm.rvs(size=10000, a=0, b=3, loc=3, scale=1)

        # Run and assert
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            distribution._fit(data)

    def test__is_constant_true(self):
        distribution = TruncatedGaussian()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = TruncatedGaussian()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()

    def test__extract_constant(self):
        distribution = TruncatedGaussian()
        distribution._params = {'a': 1, 'b': 1, 'loc': 1, 'scale': 0}

        constant = distribution._extract_constant()

        assert 1 == constant
