import warnings
from unittest import TestCase, mock

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from copulas.multivariate.gaussian import GaussianMultivariate
from tests import compare_nested_dicts


class TestGaussianCopula(TestCase):

    def setUp(self):
        """Defines random variable to use on tests. """

        self.data = pd.DataFrame({
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

    def test___init__(self):
        """On init an instance with None on all attributes except distribs is returned."""
        # Run
        copula = GaussianMultivariate()

        # Check
        assert copula.distribs == {}
        assert copula.cov_matrix is None
        assert copula.data is None
        assert copula.means is None
        assert copula.pdf is None
        assert copula.cdf is None
        assert copula.ppf is None

    def test_fit(self):
        """On fit, a distribution is created for each column along the covariance and means"""

        # Setup
        copula = GaussianMultivariate()

        # Run
        copula.fit(self.data)

        # Check
        assert copula.pdf == multivariate_normal.pdf

        for key in self.data.columns:
            assert copula.distribs[key]
            assert copula.distribs[key].mean == self.data[key].mean()
            assert copula.distribs[key].std == np.std(self.data[key])

        expected_cov, expected_means, expected_distrib = copula._get_parameters(self.data)
        assert (copula.cov_matrix == expected_cov).all().all()
        assert copula.means == expected_means
        assert (copula.distribution == expected_distrib).all().all()

    def test__get_parameters(self):
        """_get_parameters computes the covariance matrix and distribution of normalized values."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)

        expected_covariance = np.array([
            [1.04347826, -0.01316681, -0.20683455],
            [-0.01316681, 1.04347826, -0.176307],
            [-0.20683455, -0.176307, 1.04347826]
        ])
        expected_distribution = np.array([
            [1.09240792, 0.01144632, -2.03093305],
            [-0.43068285, 0.65716163, -0.51921929],
            [-1.82311642, -1.58740557, 0.44266171],
            [-0.16282474, -1.24769466, -0.39254411],
            [1.61681608, -0.33180989, 0.41696132],
            [-0.03036268, 0.58979361, 1.78229109],
            [0.48859676, -0.89581143, 0.4897179],
            [-0.2926779, 0.57063143, -0.41415579],
            [0.55055503, -1.14595689, -0.78012494],
            [0.37900618, -0.90713283, 1.30530461],
            [-0.21028524, 1.34442321, 0.1267147],
            [-0.59159617, -0.24513333, -0.26437517],
            [0.40657758, 1.50802664, -1.49016693],
            [2.03311543, -0.05566727, 0.45225054],
            [0.27622555, -1.04041319, -0.99584191],
            [-1.01710461, -0.89738776, -0.97340465],
            [-1.05206311, -0.35841816, 1.12341471],
            [-0.70629096, 1.87432672, -0.77053679],
            [0.18678009, -0.81644547, -0.38670773],
            [0.81091811, -0.06279853, 0.73843933],
            [1.63312175, 0.1583803, 0.23664886],
            [0.0204796, 2.00171183, -0.66921942],
            [-1.40626127, 1.14830217, 0.19852261],
            [-1.77133415, -0.2721289, 2.37430239]
        ])
        expected_means = expected_distribution.mean()

        # Run
        covariance, means, distribution = copula._get_parameters(self.data)

        # Check
        assert np.isclose(covariance, expected_covariance).all().all()
        assert np.isclose(distribution, expected_distribution).all().all()
        assert np.isclose(means, expected_means).all().all()

    def test_get_pdf(self):
        """get_pdf computes probability for the given values."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)
        X = np.array([[0., 0., 0.]])
        expected_result = 0.059566912334560594

        # Run
        result = copula.get_pdf(X)

        # Check
        assert result == expected_result

    def test_get_cdf(self):
        """get_cdf computes the accumulative probability values for the given values."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)
        X = np.array([1., 1., 1.])
        expected_result = 0.5822020991592192

        # Run
        result = copula.get_cdf(X)

        # Check
        assert np.isclose(result, expected_result).all().all()

    def test_get_lower_bounds(self):
        """get_lower_bounds returns the point from where cut the tail of the infinite integral."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)
        expected_result = -3.104256111232535

        # Run
        result = copula.get_lower_bound()

        # Check
        assert result == expected_result

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
