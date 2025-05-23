from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import beta

from copulas.univariate import BetaUnivariate


class TestBetaUnivariate(TestCase):
    def test__fit_constant(self):
        distribution = BetaUnivariate()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {'a': 1, 'b': 1, 'loc': 1, 'scale': 0}

    def test__fit(self):
        distribution = BetaUnivariate()

        data = beta.rvs(size=10000, a=1, b=1, loc=1, scale=1)
        distribution._fit(data)

        expected = {'loc': 1, 'scale': 1, 'a': 1, 'b': 1}
        for key, value in distribution._params.items():
            np.testing.assert_allclose(value, expected[key], atol=0.3)

    @patch('copulas.univariate.beta.beta')
    def test__fit_raises_value_error_if_scale_is_near_zero(self, mock_beta):
        mock_beta.fit.return_value = (1, 0, 1, 1e-8)
        distribution = BetaUnivariate()

        data = np.array([1, 2, 3, 4])
        err_msg = 'Converged parameters for beta distribution have a near-zero range.'
        with pytest.raises(ValueError, match=err_msg):
            distribution._fit(data)

        mock_beta.fit.assert_called_once_with(data, loc=1, scale=3)

    @patch('copulas.univariate.beta.beta')
    def test__fit_raises_value_error_if_parameters_are_higher_than_range(self, mock_beta):
        mock_beta.fit.return_value = (5, 1, 5, 1)
        distribution = BetaUnivariate()

        data = np.array([7, 8, 9, 10])
        err_msg = (
            'Converged parameters for beta distribution are outside the min/max range of the data.'
        )
        with pytest.raises(ValueError, match=err_msg):
            distribution._fit(data)

        mock_beta.fit.assert_called_once_with(data, loc=7, scale=3)

    @patch('copulas.univariate.beta.beta')
    def test__fit_raises_value_error_if_parameters_are_lower_than_range(self, mock_beta):
        mock_beta.fit.return_value = (5, 1, 5, 1)
        distribution = BetaUnivariate()

        data = np.array([1, 2, 3, 4])
        err_msg = (
            'Converged parameters for beta distribution are outside the min/max range of the data.'
        )
        with pytest.raises(ValueError, match=err_msg):
            distribution._fit(data)

        mock_beta.fit.assert_called_once_with(data, loc=1, scale=3)

    def test__is_constant_true(self):
        distribution = BetaUnivariate()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = BetaUnivariate()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()

    def test__extract_constant(self):
        distribution = BetaUnivariate()
        distribution._params = {'loc': 1, 'scale': 1, 'a': 1, 'b': 1}

        constant = distribution._extract_constant()

        assert 1 == constant

    @patch('copulas.univariate.beta.beta')
    def test__fit_loc_scale_from_beta(self, mock_beta):
        """Test that the fitted values for ``loc`` and ``scale`` are from ``scipy``.

        Test that when fitting the ``beta`` distribution, the learned ``loc`` and ``scale``
        are being the ones that the distribution returns from the ``scipy`` model.

        Setup:
            - Instanciate a ``BetaUnivariate``.

        Mock:
            - Mock the ``beta``.

        Side Effect:
            - The distribution has to learn the return values from the ``beta`` fit.
        """
        # Setup
        distribution = BetaUnivariate()
        mock_beta.fit.return_value = (1, 2, 3, 4)

        # Run
        distribution._fit(np.array([1, 2, 3, 4]))

        # Assert
        assert distribution._params == {
            'a': 1,
            'b': 2,
            'loc': 3,
            'scale': 4,
        }
