from unittest import TestCase, mock

import numpy as np
from scipy import stats

from copulas.bivariate.base import Bivariate, CopulaTypes
from tests import compare_nested_dicts


class TestBivariate(TestCase):

    def setUp(self):
        self.X = np.array([
            [2641.16233666, 180.2425623],
            [921.14476418, 192.35609972],
            [-651.32239137, 150.24830291],
            [1223.63536668, 156.62123653],
            [3233.37342355, 173.80311908],
            [1373.22400821, 191.0922843],
            [1959.28188858, 163.22252158],
            [1076.99295365, 190.73280428],
            [2029.25100261, 158.52982435],
            [1835.52188141, 163.0101334],
            [1170.03850556, 205.24904026],
            [739.42628394, 175.42916046],
            [1866.65810627, 208.31821984],
            [3703.49786503, 178.98351969],
            [1719.45232017, 160.50981075],
            [258.90206528, 163.19294974],
            [219.42363944, 173.30395132],
            [609.90212377, 215.18996298],
            [1618.44207239, 164.71141696],
            [2323.2775272, 178.84973821],
            [3251.78732274, 182.99902513],
            [1430.63989981, 217.5796917],
            [-180.57028875, 201.56983421],
            [-592.84497457, 174.92272693]
        ])

    def test_from_dict(self):
        """From_dict sets the values of a dictionary as attributes of the instance."""
        # Setup
        parameters = {
            'copula_type': 'FRANK',
            'tau': 0.15,
            'theta': 0.8
        }

        # Run
        instance = Bivariate.from_dict(parameters)

        # Check
        assert instance.copula_type == CopulaTypes.FRANK
        assert instance.tau == 0.15
        assert instance.theta == 0.8

    def test_to_dict(self):
        """To_dict returns the defining parameters of a copula in a dict."""
        # Setup
        instance = Bivariate('frank')
        instance.fit(self.X)

        expected_result = {
            'copula_type': 'FRANK',
            "tau": 0.014492753623188406,
            "theta": 0.13070829945417198
        }

        # Run
        result = instance.to_dict()

        # Check
        assert result == expected_result

    @mock.patch("builtins.open")
    @mock.patch('copulas.bivariate.base.json.dump')
    def test_save(self, json_mock, open_mock):
        """Save stores the internal dictionary as a json in a file."""
        # Setup
        instance = Bivariate('frank')
        instance.fit(self.X)

        expected_content = {
            "copula_type": "FRANK",
            "tau": 0.014492753623188406,
            "theta": 0.13070829945417198
        }

        # Run
        instance.save('test.json')

        # Check
        assert open_mock.called_once_with('test.json', 'w')
        assert json_mock.called
        compare_nested_dicts(json_mock.call_args[0][0], expected_content)

    @mock.patch('builtins.open')
    @mock.patch('copulas.bivariate.base.json.load')
    def test_load_from_file(self, json_mock, open_mock):
        """Load can recreate an instance from a saved file."""
        # Setup
        json_mock.return_value = {
            'copula_type': 'FRANK',
            'tau': -0.33333333333333337,
            'theta': -3.305771759329249
        }

        # Run
        instance = Bivariate.load('somefile.json')

        # Check
        assert open_mock.called_once_with('test.json', 'r')
        instance.copula_type == CopulaTypes.FRANK
        instance.tau == -0.33333333333333337
        instance.theta == -3.305771759329249

    def test_copula_selection_negative_tau(self):
        """If tau is negative, should choose frank copula."""
        # Setup
        X = np.array([
            [0.1, 0.6],
            [0.2, 0.5],
            [0.3, 0.4],
            [0.4, 0.3]
        ])
        assert stats.kendalltau(X[:, 0], X[:, 1])[0] < 0

        # Run
        name, param = Bivariate.select_copula(X)
        expected = CopulaTypes.FRANK

        # Check
        assert name == expected

    @mock.patch('copulas.bivariate.clayton.Clayton.partial_derivative')
    def test_partial_derivative_scalar(self, derivative_mock):
        """partial_derivative_scalar calls partial_derivative with its arguments in an array."""
        # Setup
        instance = Bivariate(CopulaTypes.CLAYTON)
        instance.fit(self.X)

        # Run
        result = instance.partial_derivative_scalar(0.5, 0.1)

        # Check
        assert result == derivative_mock.return_value

        expected_args = ((np.array([[0.5, 0.1]]), 0), {})
        assert len(expected_args) == len(derivative_mock.call_args)
        assert (derivative_mock.call_args[0][0] == expected_args[0][0]).all()
        assert derivative_mock.call_args[0][1] == expected_args[0][1]
        assert derivative_mock.call_args[1] == expected_args[1]
