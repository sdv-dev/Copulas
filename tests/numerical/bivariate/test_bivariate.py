import os
from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.bivariate import Clayton, Frank, Gumbel

DATA_DIR = os.path.join(os.path.dirname(__file__), "external")


class TestBivariateCopulas(TestCase):

    def test_densities(self):
        """
        This checks whether the pdf and cdf are correctly implemented.
        """
        df = pd.read_csv(os.path.join(DATA_DIR, "densities.csv"))
        for _, row in df.iterrows():
            copula = {
                "Clayton": Clayton(),
                "Frank": Frank(),
                "Gumbel": Gumbel()
            }[row["type"]]
            copula.theta = row["theta"]
            X = np.array([[row["x0"], row["x1"]]])
            assert np.isclose(copula.pdf(X), row["pdf"])
            assert np.isclose(copula.cdf(X), row["cdf"])

    def test_estimation(self):
        """
        This tests whether the bivariate copulas estimate the `theta` parameter
        correctly on several datasets. Note that we also check `tau` for legacy
        reasons despite the fact that it is not an actual parameter.
        """
        df = pd.read_csv(os.path.join(DATA_DIR, "estimation.csv"))
        for _, row in df.iterrows():
            copula = {
                "Clayton": Clayton(),
                "Frank": Frank(),
                "Gumbel": Gumbel()
            }[row["type"]]
            X = pd.read_csv(os.path.join(DATA_DIR, row["dataset"]))
            copula.fit(X.values)
            assert np.isclose(copula.theta, row["theta"])
            assert np.isclose(copula.tau, row["tau"])
