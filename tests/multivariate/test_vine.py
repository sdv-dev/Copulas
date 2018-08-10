from unittest import TestCase

import pandas as pd
import numpy as np

from copulas.multivariate.VineCopula import VineCopula


class TestRVineCopula(TestCase):

    def setUp(self):
        data = pd.read_csv('data/iris.data.csv')
        self.rvine = VineCopula('rvine')
        self.rvine.fit(data)

    def test_get_likelihood(self):
        uni_matrix = np.ones([1, 4])
        value = self.rvine.get_likelihood(uni_matrix)
        self.assertAlmostEquals(value, -17.328, places=3)

    def test_sample(self):
        sample_r = self.rvine.sample()
        self.assertEquals(len(sample_r), 4)
