import warnings
from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.multivariate.gaussian import GaussianMultivariate


class TestGaussianCopula(TestCase):

    def test_deprecation_warnings(self):
        """After fitting, Gaussian copula can produce new samples warningless."""
        # Setup
        copula = GaussianMultivariate()
        data = pd.read_csv('data/iris.data.csv')

        # Run
        with warnings.catch_warnings(record=True) as warns:
            copula.fit(data)
            result = copula.sample(10)

            # Check
            assert len(warns) == 0
            assert len(result) == 10

    def test_sample(self):
        """Generated samples keep the same mean and deviation as the original data."""
        copula = GaussianMultivariate()
        stats = [
            {'mean': 10000, 'std': 15},
            {'mean': 150, 'std': 10},
            {'mean': -50, 'std': 0.1}
        ]
        data = pd.DataFrame([np.random.normal(x['mean'], x['std'], 100) for x in stats]).T
        copula.fit(data)

        # Run
        result = copula.sample(1000000)

        # Check
        assert result.shape == (1000000, 3)
        for i, stat in enumerate(stats):
            expected_mean = np.mean(data[i])
            expected_std = np.std(data[i])
            result_mean = np.mean(result[i])
            result_std = np.std(result[i])

            assert abs(expected_mean - result_mean) < abs(expected_mean / 100)
            assert abs(expected_std - result_std) < abs(expected_std / 100)
