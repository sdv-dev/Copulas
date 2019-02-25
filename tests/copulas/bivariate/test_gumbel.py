from unittest import TestCase
from unittest.mock import patch

import numpy as np

from copulas.bivariate.base import Bivariate, CopulaTypes
from tests import compare_nested_iterables


class TestGumbel(TestCase):

    def setUp(self):
        self.copula = Bivariate(CopulaTypes.GUMBEL)
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

    def test_fit(self):
        """On fit, theta and tau attributes are set."""
        # Setup
        expected_theta = 1.0147
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
        expected_result = 1.003087
        X = np.array([[0.1, 0.5]])

        # Run
        result = self.copula.probability_density(X)

        # Check
        assert isinstance(result, np.ndarray)
        assert np.isclose(result, expected_result).all()

    def test_cumulative_distribution(self):
        """Cumulative_density returns the probability distribution value for a point."""
        # Setup
        self.copula.fit(self.X)
        expected_result = np.array([0.051179])
        X = np.array([[0.1, 0.5]])
        # Run

        result = self.copula.cumulative_distribution(X)

        # Check
        assert isinstance(result, np.ndarray)
        assert np.isclose(result, expected_result).all()

    @patch('copulas.bivariate.base.np.random.uniform')
    def test_sample(self, uniform_mock):
        """Sample use the inverse-transform method to generate new samples."""
        # Setup
        instance = Bivariate(CopulaTypes.GUMBEL)
        instance.tau = 0.5
        instance.theta = instance.compute_theta()

        uniform_mock.return_value = np.array([0.1, 0.2, 0.4, 0.6, 0.8])

        expected_result = np.array([
            [6.080069565509917e-06, 0.1],
            [6.080069565509917e-06, 0.2],
            [6.080069565509917e-06, 0.4],
            [6.080069565509917e-06, 0.6],
            [5.479708204503933e-06, 0.8]
        ])

        expected_uniform_call_args_list = [
            ((0, 1, 5), {}),
            ((0, 1, 5), {})
        ]

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)
        assert uniform_mock.call_args_list == expected_uniform_call_args_list

    def test_sample_random_state(self):
        """If random_state is set, the samples are the same."""
        # Setup
        instance = Bivariate(CopulaTypes.GUMBEL, random_seed=0)
        instance.tau = 0.5
        instance.theta = instance.compute_theta()

        expected_result = np.array([
            [6.08006957e-06, 5.48813504e-01],
            [6.08006957e-06, 7.15189366e-01],
            [6.59562405e-06, 6.02763376e-01],
            [3.59472685e-06, 5.44883183e-01],
            [6.08006957e-06, 4.23654799e-01]
        ])

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)
