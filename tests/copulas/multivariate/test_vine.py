from unittest import TestCase, expectedFailure

import numpy as np
import pandas as pd

from copulas.multivariate.vine import VineCopula


class TestVine(TestCase):

    def setUp(self):
        data = pd.DataFrame({
            'column1': np.array([
                2641.16233666, 921.14476418, -651.32239137, 1223.63536668,
                3233.37342355, 1373.22400821, 1959.28188858, 1076.99295365,
                2029.25100261, 1835.52188141, 1170.03850556, 739.42628394,
                1866.65810627, 3703.49786503, 1719.45232017, 258.90206528,
                219.42363944, 609.90212377, 1618.44207239, 2323.2775272,
                3251.78732274, 1430.63989981, -180.57028875, -592.84497457,
            ]),
            'column2': np.array([
                180.2425623, 192.35609972, 150.24830291, 156.62123653,
                173.80311908, 191.0922843, 163.22252158, 190.73280428,
                158.52982435, 163.0101334, 205.24904026, 175.42916046,
                208.31821984, 178.98351969, 160.50981075, 163.19294974,
                173.30395132, 215.18996298, 164.71141696, 178.84973821,
                182.99902513, 217.5796917, 201.56983421, 174.92272693
            ]),
            'column3': np.array([
                -1.42432446, -0.14759864, 0.66476302, -0.04061445, 0.64305762,
                1.79615407, 0.70450457, -0.05886671, -0.36794788, 1.39331262,
                0.39792831, 0.0676313, -0.96761759, 0.67286132, -0.55013279,
                -0.53118328, 1.23969655, -0.35985016, -0.03568531, 0.91456357,
                0.49077378, -0.27428204, 0.45857406, 2.29614033
            ])
        })

        self.rvine = VineCopula('regular')
        self.rvine.fit(data)

        self.cvine = VineCopula('center')
        self.cvine.fit(data)

        self.dvine = VineCopula('direct')
        self.dvine.fit(data)

    def test_get_likelihood(self):
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        # FIX ME: there is some randomness in rvine, will do another test
        rvalue = self.rvine.get_likelihood(uni_matrix)
        expected = -0.2859820599667698
        assert abs(rvalue - expected) < 10E-3

        cvalue = self.cvine.get_likelihood(uni_matrix)
        expected = -0.27565584158521045
        assert abs(cvalue - expected) < 10E-3

        dvalue = self.dvine.get_likelihood(uni_matrix)
        expected = -0.27565584158521045
        assert abs(dvalue - expected) < 10E-3

    @expectedFailure
    def test_sample(self):
        sample_r = self.rvine.sample()
        sample_c = self.cvine.sample()
        sample_d = self.cvine.sample()

        assert len(sample_r) == 4
        assert len(sample_c) == 4
        assert len(sample_d) == 4
