from unittest import TestCase

import numpy as np

from copulas.bivariate.independence import Independence


class TestIndependence(TestCase):

    def test___init__(self):
        """Independence copula can be instantiated directly."""
        # Setup / Run
        instance = Independence()

        # Check
        assert isinstance(instance, Independence)
        assert instance.theta is None
        assert instance.tau is None

    def test_fit(self):
        """fit checks that the given values are independent."""
        # Setup
        instance = Independence()
        data = np.array([
            [1, 2],
            [4, 3]
        ])

        # Run
        instance.fit(data)

        # Check
        instance.tau is None
        instance.theta is None

    def test_cumulative_distribution(self):
        """cumulative_distribution is the product of both probabilities."""
        # Setup
        instance = Independence()
        data = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.5, 0.5],
            [0.9, 0.9],
            [1.0, 1.0]
        ])

        expected_result = np.array([
            0.00,
            0.01,
            0.04,
            0.25,
            0.81,
            1.00,
        ])

        # Run
        result = instance.cumulative_distribution(data)

        # Check
        (result == expected_result).all().all()
