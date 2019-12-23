from unittest import TestCase
from unittest.mock import patch

import numpy as np

from copulas.bivariate.frank import Frank
from tests import compare_nested_iterables, copula_single_arg_not_one, copula_zero_if_arg_zero


class TestFrank(TestCase):

    def setUp(self):
        self.X = np.array([
            [2641.16233666, 180.2425623],
            [921.14476418, 192.35609972],
            [-651.32239137, 150.24830291],
            [1223.63536668, 156.62123653],
            [3233.37342355, 173.80311908],
            [1373.22400821, 191.0922843],
            [1959.28188858, 163.22252158],
            [1076.99295365, 190.73280428],
            [2029.25100261, 158.52982435],
            [1835.52188141, 163.0101334],
            [1170.03850556, 205.24904026],
            [739.42628394, 175.42916046],
            [1866.65810627, 208.31821984],
            [3703.49786503, 178.98351969],
            [1719.45232017, 160.50981075],
            [258.90206528, 163.19294974],
            [219.42363944, 173.30395132],
            [609.90212377, 215.18996298],
            [1618.44207239, 164.71141696],
            [2323.2775272, 178.84973821],
            [3251.78732274, 182.99902513],
            [1430.63989981, 217.5796917],
            [-180.57028875, 201.56983421],
            [-592.84497457, 174.92272693]
        ])
        self.copula = Frank()

    def test_fit(self):
        """On fit, theta and tau attributes are set."""
        # Setup
        expected_theta = 0.1307082
        expected_tau = 0.01449275

        # Run
        self.copula.fit(self.X)
        actual_theta = self.copula.theta
        actual_tau = self.copula.tau

        # Check
        self.assertAlmostEqual(actual_theta, expected_theta, places=3)
        self.assertAlmostEqual(actual_tau, expected_tau)

    def test_probability_density(self):
        """Probability_density returns the probability density for the given values."""
        # Setup
        self.copula.fit(self.X)
        expected_result = 0.999672586804842

        # Run
        result = self.copula.probability_density(np.array([[0.1, 0.5]]))

        # Check
        assert np.isclose(result, expected_result).all()
        assert isinstance(result, np.ndarray)

    def test_cumulative_distribution(self):
        """Cumulative_density returns the probability distribution value for a point."""
        # Setup
        self.copula.fit(self.X)
        expected_result = 0.05147003

        # Run
        result = self.copula.cumulative_distribution(np.array([[0.1, 0.5]]))

        # Check
        assert np.isclose(result, expected_result).all()
        assert isinstance(result, np.ndarray)

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
