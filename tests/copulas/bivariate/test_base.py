from unittest import TestCase, mock

from copulas.bivariate.base import Bivariate, CopulaTypes
from tests import compare_nested_dicts


class TestBivariate(TestCase):

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
        U = [0.1, 0.5, 0.8]
        V = [0.2, 0.7, 0.1]
        instance.fit(U, V)

        expected_result = {
            'copula_type': 'FRANK',
            'tau': -0.33333333333333337,
            'theta': -3.305771759329249
        }

        # Run
        result = instance.to_dict()

        # Check
        assert result == expected_result

    @mock.patch('copulas.bivariate.base.json.dump')
    def test_save(self, json_mock):
        """Save stores the internal dictionary as a json in a file."""
        # Setup
        instance = Bivariate('frank')
        U = [0.1, 0.5, 0.8]
        V = [0.2, 0.7, 0.1]
        instance.fit(U, V)

        expected_content = {
            "copula_type": "FRANK",
            "theta": -3.305771759329249,
            "tau": -0.33333333333333337
        }

        # Run
        instance.save('test.json')

        # Check
        assert json_mock.called
        compare_nested_dicts(json_mock.call_args[0][0], expected_content)

    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('copulas.bivariate.base.json.load')
    def test_load_from_file(self, json_mock, file_mock):
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
        instance.copula_type == CopulaTypes.FRANK
        instance.tau == -0.33333333333333337
        instance.theta == -3.305771759329249
