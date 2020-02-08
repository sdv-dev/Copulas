from unittest import TestCase
from unittest.mock import patch

import numpy as np

from copulas.bivariate.frank import Frank
from tests import compare_nested_iterables, copula_single_arg_not_one, copula_zero_if_arg_zero


class TestFrank(TestCase):

    def setUp(self):
        self.X = np.array([
            [0.2, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.4],
            [0.8, 0.6],
            [0.8, 0.9],
        ])
        self.copula = Frank()

    def test_fit(self):
        """On fit, theta and tau attributes are set."""
        self.copula.fit(self.X)
        self.assertAlmostEqual(self.copula.tau, 0.7877, places=3)
        self.assertAlmostEqual(self.copula.theta, 17.0227, places=3)

    def test_probability_density(self):
        """Probability_density returns the probability density for the given values."""
        # Setup
        self.copula.fit(self.X)
        expected_result = np.array([4.4006, 4.2302])

        # Run
        result = self.copula.probability_density(np.array([
            [0.2, 0.2],
            [0.6, 0.61]
        ]))

        # Check
        assert isinstance(result, np.ndarray)
        assert np.isclose(result, expected_result, rtol=0.05).all()

    def test_cumulative_distribution(self):
        """Cumulative_density returns the probability distribution value for a point."""
        # Setup
        self.copula.fit(self.X)
        expected_result = np.array([0.1602, 0.5641])

        # Run
        result = self.copula.cumulative_distribution(np.array([
            [0.2, 0.2],
            [0.6, 0.61]
        ]))

        # Check
        assert isinstance(result, np.ndarray)
        assert np.isclose(result, expected_result, rtol=0.05).all()

    @patch('copulas.bivariate.base.np.random.uniform')
    def test_sample(self, uniform_mock):
        """Sample use the inverse-transform method to generate new samples."""
        # Setup
        instance = Frank()
        instance.tau = 0.5
        instance.theta = instance.compute_theta()

        uniform_mock.return_value = np.array([0.1, 0.2, 0.4, 0.6, 0.8])

        expected_result = np.array([
            [0.0312640840463779, 0.1],
            [0.1007998170183327, 0.2],
            [0.3501836319841291, 0.4],
            [0.6498163680158703, 0.6],
            [0.8992001829816683, 0.8]
        ])

        expected_uniform_call_args_list = [
            ((0, 1, 5), {}),
            ((0, 1, 5), {})
        ]

        # Run
        result = instance.sample(5)

        # Check
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 2)
        compare_nested_iterables(result, expected_result)
        assert uniform_mock.call_args_list == expected_uniform_call_args_list

    def test_cdf_zero_if_single_arg_is_zero(self):
        """Test of the analytical properties of copulas on a range of values of theta."""
        # Setup
        instance = Frank()
        tau_values = np.linspace(-1.0, 1.0, 20)[1: -1]

        # Run/Check
        for tau in tau_values:
            instance.tau = tau
            instance.theta = instance.compute_theta()
            copula_zero_if_arg_zero(instance)

    def test_cdf_value_if_all_other_arg_are_one(self):
        """Test of the analytical properties of copulas on a range of values of theta."""
        # Setup
        instance = Frank()
        tau_values = np.linspace(-1.0, 1.0, 20)[1: -1]

        # Run/Check
        for tau in tau_values:
            instance.tau = tau
            instance.theta = instance.compute_theta()
            copula_single_arg_not_one(instance, tolerance=1E-03)
