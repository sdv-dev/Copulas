from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.multivariate.vine import VineCopula


class TestVine(TestCase):

    def setUp(self):
        data = pd.read_csv('data/iris.data.csv')

        self.rvine = VineCopula('rvine')
        self.rvine.fit(data)

        self.cvine = VineCopula('cvine')
        self.cvine.fit(data)

        self.dvine = VineCopula('dvine')
        self.dvine.fit(data)

    def test_get_likelihood(self):
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        # FIX ME: there is some randomness in rvine, will do another test
        rvalue = self.rvine.get_likelihood(uni_matrix)
        expected = 0
        assert abs(rvalue - expected) < 10

        cvalue = self.cvine.get_likelihood(uni_matrix)
        expected = -3.4420
        assert abs(cvalue - expected) < 10E-3

        dvalue = self.dvine.get_likelihood(uni_matrix)
        expected = -5.1838
        assert abs(dvalue - expected) < 10E-3

    # def test_sample(self):
    #     sample_r = self.rvine.sample()
    #     sample_c = self.cvine.sample()
    #     sample_d = self.cvine.sample()
    #     self.assertEquals(len(sample_r), 4)
