import warnings
from unittest import TestCase

import pandas as pd

from copulas.multivariate.GaussianCopula import GaussianCopula


class TestGaussianCopula(TestCase):

    def test_deprecation_warnings(self):
        """After fitting, Gaussian copula can produce new samples warningless."""
        # Setup
        copula = GaussianCopula()
        data = pd.read_csv('data/iris.data.csv')

        # Run
        with warnings.catch_warnings(record=True) as warns:
            copula.fit(data)
            result = copula.sample(10)

            # Check
            assert len(warns) == 0
            assert len(result) == 10
