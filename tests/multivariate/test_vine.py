from unittest import TestCase

import numpy as np
import scipy

from copulas.multivariate.VineCopula import VineCopula


class TestVine(TestCase):

    def setUp(self):
        data = '../../data/iris.data.csv'
        dvine = VineCopula('dvine')
        dvine.fit(data)

    def test_fit(self):
        
