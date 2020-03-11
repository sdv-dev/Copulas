from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from copulas.multivariate.tree import TreeTypes
from copulas.multivariate.vine import VineCopula
from tests import compare_nested_dicts, compare_nested_iterables


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

        self.rvine = VineCopula(TreeTypes.REGULAR)
        self.rvine.fit(data)

        self.cvine = VineCopula(TreeTypes.CENTER)
        self.cvine.fit(data)

        self.dvine = VineCopula(TreeTypes.DIRECT)
        self.dvine.fit(data)

    def test_get_likelihood(self):
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        # FIX ME: there is some randomness in rvine, will do another test
        rvalue = self.rvine.get_likelihood(uni_matrix)
        expected = -0.26888124854583245
        assert abs(rvalue - expected) < 10E-3

        cvalue = self.cvine.get_likelihood(uni_matrix)
        expected = -0.27565584158521045
        assert abs(cvalue - expected) < 10E-3

        dvalue = self.dvine.get_likelihood(uni_matrix)
        expected = -0.27565584158521045
        assert abs(dvalue - expected) < 10E-3

    def test_serialization_unfitted_model(self):
        """An unfitted vine can be serialized and deserialized and kept unchanged."""
        # Setup
        instance = VineCopula('regular')

        # Run
        result = VineCopula.from_dict(instance.to_dict())

        # Check
        assert result.to_dict() == instance.to_dict()

    def test_serialization_fit_model(self):
        """A fitted vine can be serialized and deserialized and kept unchanged."""
        # Setup
        instance = VineCopula('regular')
        X = pd.DataFrame(data=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        instance.fit(X)

        # Run
        result = VineCopula.from_dict(instance.to_dict())

        # Check
        compare_nested_dicts(result.to_dict(), instance.to_dict())

    @patch('copulas.multivariate.vine.np.random.randint', autospec=True)
    @patch('copulas.multivariate.vine.np.random.uniform', autospec=True)
    def test_sample_row(self, uniform_mock, randint_mock):
        """After being fit, a vine can sample new data."""
        # Setup
        instance = VineCopula(TreeTypes.REGULAR)
        X = pd.DataFrame([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], columns=list('ABCD'))
        instance.fit(X)

        uniform_mock.return_value = np.array([0.1, 0.25, 0.5, 0.75])
        randint_mock.return_value = 1
        expected_result = np.array([-0.3196499, -0.16358588, 0.418420, 1.5688347])

        # Run
        result = instance._sample_row()

        # Check
        compare_nested_iterables(result, expected_result)

        uniform_mock.assert_called_once_with(0, 1, 4)
        randint_mock.assert_called_once_with(0, 4)

    @patch('copulas.multivariate.vine.VineCopula._sample_row', autospec=True)
    def test_sample(self, sample_mock):
        """After being fit, a vine can sample new data."""
        # Setup
        vine = VineCopula(TreeTypes.REGULAR)
        X = pd.DataFrame([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], columns=list('ABCD'))
        vine.fit(X)

        expected_result = pd.DataFrame([
            {'A': 1, 'B': 2, 'C': 3, 'D': 4},
            {'A': 1, 'B': 2, 'C': 3, 'D': 4},
            {'A': 1, 'B': 2, 'C': 3, 'D': 4},
            {'A': 1, 'B': 2, 'C': 3, 'D': 4},
            {'A': 1, 'B': 2, 'C': 3, 'D': 4},
        ])

        sample_mock.return_value = np.array([1, 2, 3, 4])

        # Run
        result = vine.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

        assert sample_mock.call_count == 5

    def test_sample_random_state(self):
        """When random_state is set, the generated samples are always the same."""
        # Setup
        vine = VineCopula(TreeTypes.REGULAR, random_seed=0)
        X = pd.DataFrame([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        vine.fit(X)

        expected_result = pd.DataFrame(
            [[0.101933, 0.527734, 0.080266, 0.078328]],
            columns=range(4)
        )

        # Run
        result = vine.sample(1)

        # Check
        compare_nested_iterables(result, expected_result)
