import warnings
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from copulas import get_qualified_name
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

    def test___init__default_args(self):
        """On init an instance with None on all attributes except distribs is returned."""
        # Run
        copula = GaussianMultivariate()

        # Check
        assert copula.distribs == {}
        assert copula.covariance is None
        assert copula.means is None
        assert copula.distribution == 'copulas.univariate.gaussian.GaussianUnivariate'

    def test__init__distribution_arg(self):
        """On init the distribution argument is set as attribute."""
        # Setup
        distribution = 'full.qualified.name.of.distribution'

        # Run
        copula = GaussianMultivariate(distribution)

        # Check
        assert copula.distribs == {}
        assert copula.covariance is None
        assert copula.means is None
        assert copula.distribution == 'full.qualified.name.of.distribution'

    def test_fit_default_distribution(self):
        """On fit, a distribution is created for each column along the covariance and means"""

        # Setup
        copula = GaussianMultivariate()

        # Run
        copula.fit(self.data)

        # Check
        assert copula.distribution == 'copulas.univariate.gaussian.GaussianUnivariate'

        for key in self.data.columns:
            assert key in copula.distribs
            assert get_qualified_name(copula.distribs[key].__class__) == copula.distribution
            assert copula.distribs[key].mean == self.data[key].mean()
            assert copula.distribs[key].std == np.std(self.data[key])

        expected_covariance = copula._get_covariance(self.data)
        assert (copula.covariance == expected_covariance).all().all()

    def test_fit_distribution_arg(self):
        """On fit, the distributions for each column use instances of copula.distribution."""
        # Setup
        distribution = 'copulas.univariate.gaussian_kde.GaussianKDE'
        copula = GaussianMultivariate(distribution=distribution)

        # Run
        copula.fit(self.data)

        # Check
        assert copula.distribution == 'copulas.univariate.gaussian_kde.GaussianKDE'

        for key in self.data.columns:
            assert key in copula.distribs
            assert get_qualified_name(copula.distribs[key].__class__) == copula.distribution

        expected_covariance = copula._get_covariance(self.data)
        assert (copula.covariance == expected_covariance).all().all()

    def test_fit_numpy_array(self):
        """Fit should work indistinctly with numpy arrays and pandas dataframes """
        # Setup
        copula = GaussianMultivariate()

        # Run
        copula.fit(self.data.values)

        # Check
        for key, column in enumerate(self.data.columns):
            assert copula.distribs[key]
            assert copula.distribs[key].mean == np.mean(self.data[column])
            assert copula.distribs[key].std == np.std(self.data[column])

        expected_covariance = copula._get_covariance(pd.DataFrame(self.data.values))
        assert (copula.covariance == expected_covariance).all().all()

    def test__get_covariance(self):
        """_get_covariance computes the covariance matrix of normalized values."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)

        expected_covariance = np.array([
            [1.04347826, -0.01316681, -0.20683455],
            [-0.01316681, 1.04347826, -0.176307],
            [-0.20683455, -0.176307, 1.04347826]
        ])

        # Run
        covariance = copula._get_covariance(self.data)

        # Check
        assert np.isclose(covariance, expected_covariance).all().all()

    def test_probability_density(self):
        """Probability_density computes probability for the given values."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)
        X = np.array([[0., 0., 0.]])
        expected_result = 0.059566912334560594

        # Run
        result = copula.probability_density(X)

        # Check
        assert result == expected_result

    def test_cumulative_distribution_fit_df_call_np_array(self):
        """Cumulative_density integrates the probability density along the given values."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)
        X = np.array([1., 1., 1.])
        expected_result = 0.5822020991592192

        # Run
        result = copula.cumulative_distribution(X)

        # Check
        assert np.isclose(result, expected_result).all().all()

    def test_cumulative_distribution_fit_call_np_array(self):
        """Cumulative_density integrates the probability density along the given values."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data.values)
        X = np.array([1., 1., 1.])
        expected_result = 0.5822020991592192

        # Run
        result = copula.cumulative_distribution(X)

        # Check
        assert np.isclose(result, expected_result).all().all()

    def test_cumulative_distribution_fit_call_pd(self):
        """Cumulative_density integrates the probability density along the given values."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data.values)
        X = pd.Series([1., 1., 1.])
        expected_result = 0.5822020991592192

        # Run
        result = copula.cumulative_distribution(X)

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

    @patch('copulas.multivariate.gaussian.np.random.multivariate_normal')
    def test_sample(self, normal_mock):
        """Sample use the inverse-transform method to generate new samples."""
        # Setup
        instance = GaussianMultivariate()
        data = pd.DataFrame([
            {'A': 25, 'B': 75, 'C': 100},
            {'A': 30, 'B': 60, 'C': 250},
            {'A': 10, 'B': 65, 'C': 350},
            {'A': 20, 'B': 80, 'C': 150},
            {'A': 25, 'B': 70, 'C': 500}
        ])
        instance.fit(data)

        normal_mock.return_value = np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8]
        ])

        expected_result = pd.DataFrame([
            {'A': 22.678232998312527, 'B': 70.70710678118655, 'C': 284.35270009440734},
            {'A': 23.356465996625055, 'B': 71.41421356237309, 'C': 298.7054001888146},
            {'A': 24.712931993250110, 'B': 72.82842712474618, 'C': 327.4108003776293},
            {'A': 26.069397989875164, 'B': 74.24264068711929, 'C': 356.116200566444},
            {'A': 27.425863986500215, 'B': 75.65685424949238, 'C': 384.8216007552586}
        ])

        # Run
        result = instance.sample(5)

        # Check
        assert result.equals(expected_result)

        assert normal_mock.called_once_with(
            np.zeros(instance.covariance.shape[0]),
            instance.covariance,
            5
        )

    def test_sample_random_state(self):
        """When random_state is set the samples are the same."""
        # Setup
        instance = GaussianMultivariate(random_seed=0)
        data = pd.DataFrame([
            {'A': 25, 'B': 75, 'C': 100},
            {'A': 30, 'B': 60, 'C': 250},
            {'A': 10, 'B': 65, 'C': 350},
            {'A': 20, 'B': 80, 'C': 150},
            {'A': 25, 'B': 70, 'C': 500}
        ])
        instance.fit(data)

        expected_result = pd.DataFrame([
            {'A': 25.566882482769294, 'B': 61.01690157277244, 'C': 575.71068885087790},
            {'A': 32.624255560452110, 'B': 47.31477394460025, 'C': 447.84049148268970},
            {'A': 20.117642182744806, 'B': 63.68224998298797, 'C': 397.76402526341593},
            {'A': 25.357483201156676, 'B': 72.30337152729443, 'C': 433.06766240515134},
            {'A': 23.202174689737113, 'B': 66.32056962524452, 'C': 405.08384853948280}
        ])

        # Run
        result = instance.sample(5)

        # Check
        assert result.equals(expected_result)

    def test_to_dict(self):
        """To_dict returns the parameters to replicate the copula."""
        # Setup
        copula = GaussianMultivariate()
        data = pd.read_csv('data/iris.data.csv')
        copula.fit(data)
        covariance = [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]
        expected_result = {
            'covariance': covariance,
            'fitted': True,
            'type': 'copulas.multivariate.gaussian.GaussianMultivariate',
            'distribution': 'copulas.univariate.gaussian.GaussianUnivariate',
            'distribs': {
                'feature_01': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 5.843333333333334,
                    'std': 0.8253012917851409,
                    'fitted': True,
                },
                'feature_02': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 3.0540000000000003,
                    'std': 0.4321465800705435,
                    'fitted': True,
                },
                'feature_03': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 3.758666666666666,
                    'std': 1.7585291834055212,
                    'fitted': True,
                },
                'feature_04': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 1.1986666666666668,
                    'std': 0.7606126185881716,
                    'fitted': True,
                }
            }
        }

        # Run
        result = copula.to_dict()

        # Check
        compare_nested_dicts(result, expected_result)

    def test_from_dict(self):
        """from_dict generates a new instance from its parameters."""
        # Setup
        covariance = [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]
        parameters = {
            'covariance': covariance,
            'fitted': True,
            'type': 'copulas.multivariate.gaussian.GaussianMultivariate',
            'distribution': 'copulas.univariate.gaussian.GaussianUnivariate',
            'distribs': {
                'feature_01': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 5.843333333333334,
                    'std': 0.8253012917851409,
                    'fitted': True,
                },
                'feature_02': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 3.0540000000000003,
                    'std': 0.4321465800705435,
                    'fitted': True,
                },
                'feature_03': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 3.758666666666666,
                    'std': 1.7585291834055212,
                    'fitted': True,
                },
                'feature_04': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 1.1986666666666668,
                    'std': 0.7606126185881716,
                    'fitted': True,
                }
            }
        }

        # Run
        copula = GaussianMultivariate.from_dict(parameters)

        # Check
        assert (copula.covariance == covariance).all()

        for name, distrib in copula.distribs.items():
            assert copula.distribs[name].to_dict() == parameters['distribs'][name]

        # This isn't to check the sampling, but that the copula is able to run.
        assert copula.sample(10).all().all()

    @patch("builtins.open")
    @patch('copulas.multivariate.base.json.dump')
    def test_save(self, json_mock, open_mock):
        """Save stores the internal dictionary as a json in a file."""
        # Setup
        instance = GaussianMultivariate()
        data = pd.read_csv('data/iris.data.csv')
        instance.fit(data)
        covariance = [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]
        expected_content = {
            'covariance': covariance,
            'fitted': True,
            'type': 'copulas.multivariate.gaussian.GaussianMultivariate',
            'distribution': 'copulas.univariate.gaussian.GaussianUnivariate',
            'distribs': {
                'feature_01': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 5.843333333333334,
                    'std': 0.8253012917851409,
                    'fitted': True,
                },
                'feature_02': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 3.0540000000000003,
                    'std': 0.4321465800705435,
                    'fitted': True,
                },
                'feature_03': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 3.758666666666666,
                    'std': 1.7585291834055212,
                    'fitted': True,
                },
                'feature_04': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 1.1986666666666668,
                    'std': 0.7606126185881716,
                    'fitted': True,
                }
            }
        }

        # Run
        instance.save('test.json')

        # Check
        assert open_mock.called_once_with('test.json', 'w')
        compare_nested_dicts(json_mock.call_args[0][0], expected_content)

    @patch('builtins.open')
    @patch('copulas.bivariate.base.json.load')
    def test_load(self, json_mock, open_mock):
        """Load can recreate an instance from a saved file."""
        # Setup
        covariance = [
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ]
        json_mock.return_value = {
            'covariance': covariance,
            'fitted': True,
            'type': 'copulas.multivariate.gaussian.GaussianMultivariate',
            'distribution': 'copulas.univariate.gaussian.GaussianUnivariate',
            'distribs': {
                'feature_01': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 5.843333333333334,
                    'std': 0.8253012917851409,
                    'fitted': True,
                },
                'feature_02': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 3.0540000000000003,
                    'std': 0.4321465800705435,
                    'fitted': True,
                },
                'feature_03': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 3.758666666666666,
                    'std': 1.7585291834055212,
                    'fitted': True,
                },
                'feature_04': {
                    'type': 'copulas.univariate.gaussian.GaussianUnivariate',
                    'mean': 1.1986666666666668,
                    'std': 0.7606126185881716,
                    'fitted': True,
                }
            }
        }

        # Run
        instance = GaussianMultivariate.load('test.json')

        # Check
        assert (instance.covariance == np.array([
            [1.006711409395973, -0.11010327176239865, 0.8776048563471857, 0.823443255069628],
            [-0.11010327176239865, 1.006711409395972, -0.4233383520816991, -0.3589370029669186],
            [0.8776048563471857, -0.4233383520816991, 1.006711409395973, 0.9692185540781536],
            [0.823443255069628, -0.3589370029669186, 0.9692185540781536, 1.0067114093959735]
        ])).all()

        for name, distrib in instance.distribs.items():
            assert instance.distribs[name].to_dict() == json_mock.return_value['distribs'][name]

        assert open_mock.called_once_with('test.json', 'r')

    def test_sample_constant_column(self):
        """Gaussian copula can sample after being fit with a constant column.

        This process will raise warnings when computing the covariance matrix
        """
        # Setup
        instance = GaussianMultivariate()
        X = np.array([
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5]
        ])
        instance.fit(X)

        # Run
        result = instance.sample(5)

        # Check
        assert result.shape == (5, 2)
        assert result[~result.isnull()].all().all()
        assert result.loc[:, 0].equals(pd.Series([1, 1, 1, 1, 1], name=0))

        # This is to check that the samples on the non constant column are not constant too.
        assert len(result.loc[:, 1].unique()) > 1

        covariance = instance.covariance
        assert (~pd.isnull(covariance)).all().all()
