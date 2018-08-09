from unittest import TestCase

import pandas as pd

from copulas.multivariate.vine import VineCopula


class TestVineCopula(TestCase):

    def setUp(self):
        data = pd.read_csv('data/iris.data.csv')
        self.dvine = VineCopula('dvine')
        self.dvine.fit(data)

        self.cvine = VineCopula('rvine')
        self.cvine.fit(data)

        self.rvine = VineCopula('rvine')
        self.rvine.fit(data)

    def test_sample(self):
        sample_r = self.rvine.sample()
        sample_d = self.dvine.sample()
        sample_c = self.cvine.sample()

        self.assertEquals(len(sample_r), 4)
        self.assertEquals(len(sample_d), 4)
        self.assertEquals(len(sample_c), 4)
