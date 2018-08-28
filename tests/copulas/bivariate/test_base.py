import json
from unittest import TestCase, mock

from copulas.bivariate.base import Bivariate, CopulaTypes
from tests import compare_nested_dicts


class TestBivariate(TestCase):

    def test_from_dict(self):
        """From_dict sets the values of a dictionary as attributes of the instance."""
        # Setup
        instance = Bivariate('frank')
        parameters = {
            'tau': 0.15,
            'theta': 0.8
        }

        # Run
        instance.from_dict(**parameters)

        # Check
        assert instance.tau == 0.15
        assert instance.theta == 0.8

    def test_to_dict(self):
        """To_dict returns the defining parameters of a copula in a dict."""
        # Setup
        instance = Bivariate('frank')
        U = [0.1, 0.5, 0.8]
        V = [0.2, 0.7, 0.1]
        instance.fit(U, V)

        expected_result = {
            'class': 'FRANK',
            'tau': -0.33333333333333337,
            'theta': -3.305771759329249
        }

        # Run
        result = instance.to_dict()

        # Check
        assert result == expected_result

    @mock.patch('builtins.open', new_callable=mock.mock_open)
    def test_save(self, file_mock):
        """Save stores the internal dictionary as a json in a file."""
        # Setup
        instance = Bivariate('frank')
        U = [0.1, 0.5, 0.8]
        V = [0.2, 0.7, 0.1]
        instance.fit(U, V)

        expected_content = {
            "class": "FRANK",
            "theta": -3.305771759329249,
            "tau": -0.33333333333333337
        }

        # Run
        instance.save('test.json')

        # Check
        file_mock.assert_called_once_with('test.json', 'w')  # Opening of the file
        write_mock = file_mock.return_value.write
        assert write_mock.call_count == 1
        compare_nested_dicts(json.loads(write_mock.call_args[0][0]), expected_content)

    @mock.patch('copulas.bivariate.base.json.loads')
    def test_load_from_file(self, json_mock):
        """Load can recreate an instance from a saved file."""
        # Setup
        json_mock.return_value = {
            'class': 'FRANK',
            'tau': -0.33333333333333337,
            'theta': -3.305771759329249
        }

        # Run
        instance = Bivariate.load('somefile.json')

        # Check
        instance.copula_type == CopulaTypes.FRANK
        instance.tau == -0.33333333333333337
        instance.theta == -3.305771759329249

    def test_load_from_dict(self):
        """Load can recreate an instance from an dict containing the parameters."""
        # Setup
        parameters = {
            'class': 'FRANK',
            'tau': -0.33333333333333337,
            'theta': -3.305771759329249
        }

        # Run
        instance = Bivariate.load(parameters)

        # Check
        instance.copula_type == CopulaTypes.FRANK
        instance.tau == -0.33333333333333337
        instance.theta == -3.305771759329249
