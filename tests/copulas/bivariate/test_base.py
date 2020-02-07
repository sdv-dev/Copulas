from unittest import TestCase, mock

import numpy as np

from copulas.bivariate.base import Bivariate, CopulaTypes
from tests import compare_nested_dicts


class TestBivariate(TestCase):

    def setUp(self):
        self.X = np.array([
            [0.2, 0.3],
            [0.4, 0.4],
            [0.6, 0.4],
            [0.8, 0.6],
        ])

    def test___init__random_seed(self):
        """If random_seed is passed as argument, will be set as attribute."""
        # Setup
        random_seed = 'random_seed'

        # Run
        instance = Bivariate(copula_type=CopulaTypes.CLAYTON, random_seed=random_seed)

        # Check
        assert instance.random_seed == 'random_seed'

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
        instance = Bivariate(copula_type='frank')
        instance.fit(self.X)

        expected_result = {
            'copula_type': 'FRANK',
            "tau": 0.9128709291752769,
            "theta": 44.2003852484162
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
        instance = Bivariate(copula_type='frank')
        instance.fit(self.X)

        expected_content = {
            "copula_type": "FRANK",
            "tau": 0.9128709291752769,
            "theta": 44.2003852484162
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

    @mock.patch('copulas.bivariate.clayton.Clayton.partial_derivative')
    def test_partial_derivative_scalar(self, derivative_mock):
        """partial_derivative_scalar calls partial_derivative with its arguments in an array."""
        # Setup
        instance = Bivariate(copula_type=CopulaTypes.CLAYTON)
        instance.fit(self.X)

        # Run
        result = instance.partial_derivative_scalar(0.5, 0.1)

        # Check
        assert result == derivative_mock.return_value

        expected_args = ((np.array([[0.5, 0.1]]), 0), {})
        assert len(expected_args) == len(derivative_mock.call_args)
        assert (derivative_mock.call_args[0][0] == expected_args[0][0]).all()
