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
        copula = GaussianMultivariate()
        data = pd.DataFrame([[-1, 0, 1], [1, -1, 0], [0, 1, -1]])
        copula.fit(data)

        # Run
        result = copula.sample(1000000)

        # Check
        assert len(result) == 1000000
        for i in range(result.shape[1]):
            data_mean = np.mean(data.loc[:, i])
            result_mean = np.mean(result.loc[:, i])
            assert abs(data_mean - result_mean) < 10E-3
            assert abs(data_mean - result_mean) < 10E-3
