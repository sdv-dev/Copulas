from unittest import TestCase

import numpy as np

from copulas.bivariate.clayton import Clayton
from tests import copula_single_arg_not_one, copula_zero_if_arg_zero


class TestClayton(TestCase):

    def setUp(self):
        self.copula = Clayton()
        self.X = np.array([
            [0.2, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.4],
            [0.8, 0.6],
            [0.8, 0.9],
        ])

    def test_fit(self):
        """On fit, theta and tau attributes are set."""
        self.copula.fit(self.X)
        self.assertAlmostEqual(self.copula.tau, 0.7877, places=3)
        self.assertAlmostEqual(self.copula.theta, 7.4218, places=3)

    def test_probability_density(self):
        """Probability_density returns the probability density for the given values."""
        # Setup
        self.copula.fit(self.X)
        expected_result = np.array([9.5886, 3.2394])

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
        expected_result = np.array([0.1821, 0.5517])

        # Run
        result = self.copula.cumulative_distribution(np.array([
            [0.2, 0.2],
            [0.6, 0.61]
        ]))

        # Check
        assert isinstance(result, np.ndarray)
        assert np.isclose(result, expected_result, rtol=0.05).all()

    def test_partial_derivative(self):
        """Probability_density returns the probability density for the given values."""
        self.copula.fit(self.X)
        U = np.array([0.1, 0.2, 0.3, 0.5])
        V = np.array([0.3, 0.5, 0.6, 0.5])

        # Direct implementation of the derivative
        result1 = self.copula.partial_derivative(np.column_stack((U, V)))

        # Finite difference implementation of the derivative
        result2 = super(Clayton, self.copula).partial_derivative(np.column_stack((U, V)))

        assert np.isclose(result1, result2, rtol=0.01).all()

    def test_inverse_cumulative_percentile_point(self):
        """The percentile point and cumulative_distribution should be inverse one of the other."""
        self.copula.fit(self.X)

        U = np.array([0.1, 0.2, 0.3])
        V = np.array([0.3, 0.5, 0.6])
        cdf_percentile = self.copula.partial_derivative(np.column_stack((U, V)))
        U_inferred = self.copula.percent_point(cdf_percentile, V)

        assert np.isclose(U, U_inferred).all()

    def test_cdf_zero_if_single_arg_is_zero(self):
        """Test of the analytical properties of copulas on a range of values of theta."""
        # Setup
        instance = Clayton()
        tau_values = np.linspace(0.0, 1.0, 20)[1: -1]

        # Run/Check
        for tau in tau_values:
            instance.tau = tau
            instance.theta = instance.compute_theta()
            copula_zero_if_arg_zero(instance)

    def test_cdf_value_if_all_other_arg_are_one(self):
        """Test of the analytical properties of copulas on a range of values of theta."""
        # Setup
        instance = Clayton()
        tau_values = np.linspace(0.0, 1.0, 20)[1: -1]

        # Run/Check
        for tau in tau_values:
            instance.tau = tau
            instance.theta = instance.compute_theta()
            copula_single_arg_not_one(instance)
