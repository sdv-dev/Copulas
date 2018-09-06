import warnings
from unittest import TestCase, mock

import numpy as np
import pandas as pd

from copulas.multivariate.gaussian import GaussianMultivariate
from tests import compare_nested_dicts


class TestGaussianCopula(TestCase):

    def test_deprecation_warnings(self):
        """After fitting, Gaussian copula can produce new samples warningless."""
        # Setup
        copula = GaussianMultivariate()
        data = pd.read_csv('data/iris.data.csv')

        # Run
        with warnings.catch_warnings(record=True) as warns:
            copula.fit(data)
            result = copula.sample(10)

            # Check
            assert len(warns) == 0
            assert len(result) == 10

    def test_sample(self):
        """Generated samples keep the same mean and deviation as the original data."""
        copula = GaussianMultivariate()
        stats = [
            {'mean': 10000, 'std': 15},
            {'mean': 150, 'std': 10},
            {'mean': -50, 'std': 0.1}
        ]
        data = pd.DataFrame([np.random.normal(x['mean'], x['std'], 100) for x in stats]).T
        copula.fit(data)

        # Run
        result = copula.sample(1000000)

        # Check
        assert result.shape == (1000000, 3)
        for i, stat in enumerate(stats):
            expected_mean = np.mean(data[i])
            expected_std = np.std(data[i])
            result_mean = np.mean(result[i])
            result_std = np.std(result[i])

            assert abs(expected_mean - result_mean) < abs(expected_mean / 100)
            assert abs(expected_std - result_std) < abs(expected_std / 100)

    def test_to_dict(self):
        """To_dict returns the parameters to replicate the copula."""
        # Setup
        copula = GaussianMultivariate()
        data = pd.read_csv('data/iris.data.csv')
        copula.fit(data)
        cov_matrix = [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]
        expected_result = {
            'means': [
                -3.315866100213801e-16,
                -7.815970093361102e-16,
                2.842170943040401e-16,
                -2.3684757858670006e-16
            ],
            'cov_matrix': cov_matrix,
            'distribs': {
                'feature_01': {'mean': 5.843333333333334, 'std': 0.8253012917851409},
                'feature_02': {'mean': 3.0540000000000003, 'std': 0.4321465800705435},
                'feature_03': {'mean': 3.758666666666666, 'std': 1.7585291834055212},
                'feature_04': {'mean': 1.1986666666666668, 'std': 0.7606126185881716}
            }
        }

        # Run
        result = copula.to_dict()

        # Check
        compare_nested_dicts(result, expected_result)

    def test_from_dict(self):
        """ """
        # Setup
        cov_matrix = [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]
        parameters = {
            'means': [
                -3.315866100213801e-16,
                -7.815970093361102e-16,
                2.842170943040401e-16,
                -2.3684757858670006e-16
            ],
            'cov_matrix': cov_matrix,
            'distribs': {
                'feature_01': {'mean': 5.843333333333334, 'std': 0.8253012917851409},
                'feature_02': {'mean': 3.0540000000000003, 'std': 0.4321465800705435},
                'feature_03': {'mean': 3.758666666666666, 'std': 1.7585291834055212},
                'feature_04': {'mean': 1.1986666666666668, 'std': 0.7606126185881716}
            }
        }

        # Run
        copula = GaussianMultivariate.from_dict(parameters)

        # Check
        assert copula.means == [
            -3.315866100213801e-16,
            -7.815970093361102e-16,
            2.842170943040401e-16,
            -2.3684757858670006e-16
        ]
        assert (copula.cov_matrix == [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]).all()
        for name, distrib in copula.distribs.items():
            assert copula.distribs[name].to_dict() == parameters['distribs'][name]

        # This isn't to check the sampling, but that the copula is able to run.
        assert copula.sample(10).all().all()

    @mock.patch('copulas.multivariate.base.json.dump')
    def test_save(self, json_mock):
        """Save stores the internal dictionary as a json in a file."""
        # Setup
        instance = GaussianMultivariate()
        data = pd.read_csv('data/iris.data.csv')
        instance.fit(data)
        cov_matrix = [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]
        parameters = {
            'means': [
                -3.315866100213801e-16,
                -7.815970093361102e-16,
                2.842170943040401e-16,
                -2.3684757858670006e-16
            ],
            'cov_matrix': cov_matrix,
            'distribs': {
                'feature_01': {'mean': 5.843333333333334, 'std': 0.8253012917851409},
                'feature_02': {'mean': 3.0540000000000003, 'std': 0.4321465800705435},
                'feature_03': {'mean': 3.758666666666666, 'std': 1.7585291834055212},
                'feature_04': {'mean': 1.1986666666666668, 'std': 0.7606126185881716}
            }
        }
        expected_content = parameters

        # Run
        instance.save('test.json')

        # Check
        compare_nested_dicts(json_mock.call_args[0][0], expected_content)

    @mock.patch('builtins.open', new_callable=mock.mock_open)
    @mock.patch('copulas.bivariate.base.json.load')
    def test_load(self, json_mock, file_mock):
        """Load can recreate an instance from a saved file."""
        # Setup
        cov_matrix = [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]
        json_mock.return_value = {
            'means': [
                -3.315866100213801e-16,
                -7.815970093361102e-16,
                2.842170943040401e-16,
                -2.3684757858670006e-16
            ],
            'cov_matrix': cov_matrix,
            'distribs': {
                'feature_01': {'mean': 5.843333333333334, 'std': 0.8253012917851409},
                'feature_02': {'mean': 3.0540000000000003, 'std': 0.4321465800705435},
                'feature_03': {'mean': 3.758666666666666, 'std': 1.7585291834055212},
                'feature_04': {'mean': 1.1986666666666668, 'std': 0.7606126185881716}
            }
        }

        # Run
        instance = GaussianMultivariate.load('somefile.json')

        # Check
        assert instance.means == [
            -3.315866100213801e-16,
            -7.815970093361102e-16,
            2.842170943040401e-16,
            -2.3684757858670006e-16
        ]

        assert (instance.cov_matrix == [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]).all()

        for name, distrib in instance.distribs.items():
            assert instance.distribs[name].to_dict() == json_mock.return_value['distribs'][name]
