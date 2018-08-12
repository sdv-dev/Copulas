import logging
from unittest import TestCase

import pandas as pd
import numpy as np

from copulas.multivariate.vine import VineCopula

LOGGER = logging.getLogger(__name__)


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

        rvalue = self.rvine.get_likelihood(uni_matrix)
        # self.assertAlmostEquals(rvalue, 12.8889, places=3)

        cvalue = self.cvine.get_likelihood(uni_matrix)
        self.assertAlmostEquals(cvalue,  -3.0428, places=3)

        dvalue = self.dvine.get_likelihood(uni_matrix)
        self.assertAlmostEquals(dvalue, -3.8156, places=3)

    # def test_sample(self):
    #     sample_r = self.rvine.sample()
    #     sample_c = self.cvine.sample()
    #     sample_d = self.cvine.sample()
    #     self.assertEquals(len(sample_r), 4)
