from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from copulas import get_qualified_name
from copulas.multivariate.gaussian import GaussianMultivariate
from copulas.univariate import GaussianUnivariate


class TestGaussianMultivariate(TestCase):

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

    def test__transform_to_normal_numpy_1d(self):
        # Setup
        gm = GaussianMultivariate()
        dist_a = Mock()
        dist_a.cdf.return_value = np.array([0])
        dist_b = Mock()
        dist_b.cdf.return_value = np.array([0.3])
        gm.columns = ['a', 'b']
        gm.univariates = [dist_a, dist_b]

        # Run
        data = np.array([
            [3, 5],
        ])
        returned = gm._transform_to_normal(data)

        # Check
        # Failures may occurr on different cpytonn implementations
        # with different float precision values.
        # If that happens, atol might need to be increased
        expected = np.array([
            [-5.166579, -0.524401],
        ])
        np.testing.assert_allclose(returned, expected, atol=1e-6)

        assert dist_a.cdf.call_count == 1
        expected = np.array([3])
        passed = dist_a.cdf.call_args[0][0]
        np.testing.assert_allclose(expected, passed)

        assert dist_b.cdf.call_count == 1
        expected = np.array([5])
        passed = dist_b.cdf.call_args[0][0]
        np.testing.assert_allclose(expected, passed)

    def test__transform_to_normal_numpy_2d(self):
        # Setup
        gm = GaussianMultivariate()
        dist_a = Mock()
        dist_a.cdf.return_value = np.array([0, 0.5, 1])
        dist_b = Mock()
        dist_b.cdf.return_value = np.array([0.3, 0.5, 0.7])
        gm.columns = ['a', 'b']
        gm.univariates = [dist_a, dist_b]

        # Run
        data = np.array([
            [3, 5],
            [4, 6],
            [5, 7],
        ])
        returned = gm._transform_to_normal(data)

        # Check
        # Failures may occurr on different cpytonn implementations
        # with different float precision values.
        # If that happens, atol might need to be increased
        expected = np.array([
            [-5.166579, -0.524401],
            [0.0, 0.0],
            [5.166579, 0.524401]
        ])
        np.testing.assert_allclose(returned, expected, atol=1e-6)

        assert dist_a.cdf.call_count == 1
        expected = np.array([3, 4, 5])
        passed = dist_a.cdf.call_args[0][0]
        np.testing.assert_allclose(expected, passed)

        assert dist_b.cdf.call_count == 1
        expected = np.array([5, 6, 7])
        passed = dist_b.cdf.call_args[0][0]
        np.testing.assert_allclose(expected, passed)

    def test__transform_to_normal_series(self):
        # Setup
        gm = GaussianMultivariate()
        dist_a = Mock()
        dist_a.cdf.return_value = np.array([0])
        dist_b = Mock()
        dist_b.cdf.return_value = np.array([0.3])
        gm.columns = ['a', 'b']
        gm.univariates = [dist_a, dist_b]

        # Run
        data = pd.Series({'a': 3, 'b': 5})
        returned = gm._transform_to_normal(data)

        # Check
        # Failures may occurr on different cpytonn implementations
        # with different float precision values.
        # If that happens, atol might need to be increased
        expected = np.array([
            [-5.166579, -0.524401],
        ])
        np.testing.assert_allclose(returned, expected, atol=1e-6)

        assert dist_a.cdf.call_count == 1
        expected = np.array([3])
        passed = dist_a.cdf.call_args[0][0]
        np.testing.assert_allclose(expected, passed)

        assert dist_b.cdf.call_count == 1
        expected = np.array([5])
        passed = dist_b.cdf.call_args[0][0]
        np.testing.assert_allclose(expected, passed)

    def test__transform_to_normal_dataframe(self):
        # Setup
        gm = GaussianMultivariate()
        dist_a = Mock()
        dist_a.cdf.return_value = np.array([0, 0.5, 1])
        dist_b = Mock()
        dist_b.cdf.return_value = np.array([0.3, 0.5, 0.7])
        gm.columns = ['a', 'b']
        gm.univariates = [dist_a, dist_b]

        # Run
        data = pd.DataFrame({
            'a': [3, 4, 5],
            'b': [5, 6, 7]
        })
        returned = gm._transform_to_normal(data)

        # Check
        # Failures may occurr on different cpytonn implementations
        # with different float precision values.
        # If that happens, atol might need to be increased
        expected = np.array([
            [-5.166579, -0.524401],
            [0.0, 0.0],
            [5.166579, 0.524401]
        ])
        np.testing.assert_allclose(returned, expected, atol=1e-6)

        assert dist_a.cdf.call_count == 1
        expected = np.array([3, 4, 5])
        passed = dist_a.cdf.call_args[0][0]
        np.testing.assert_allclose(expected, passed)

        assert dist_b.cdf.call_count == 1
        expected = np.array([5, 6, 7])
        passed = dist_b.cdf.call_args[0][0]
        np.testing.assert_allclose(expected, passed)

    def test__get_covariance(self):
        """_get_covariance computes the covariance matrix of normalized values."""
        # Setup
        copula = GaussianMultivariate(GaussianUnivariate)
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

    def test_fit_default_distribution(self):
        """On fit, a distribution is created for each column along the covariance and means"""

        copula = GaussianMultivariate(GaussianUnivariate)
        copula.fit(self.data)

        for i, key in enumerate(self.data.columns):
            assert copula.columns[i] == key
            assert copula.univariates[i].__class__ == GaussianUnivariate
            assert copula.univariates[i]._params['loc'] == self.data[key].mean()
            assert copula.univariates[i]._params['scale'] == np.std(self.data[key])

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

        for i, key in enumerate(self.data.columns):
            assert copula.columns[i] == key
            assert get_qualified_name(copula.univariates[i].__class__) == copula.distribution

        expected_covariance = copula._get_covariance(self.data)
        assert (copula.covariance == expected_covariance).all().all()

    def test_fit_distribution_selector(self):
        """
        On fit, it should use the correct distributions for those that are
        specified and default to using the base class otherwise.
        """
        copula = GaussianMultivariate(distribution={
            'column1': 'copulas.univariate.beta.BetaUnivariate',
            'column2': 'copulas.univariate.gaussian_kde.GaussianKDE',
        })
        copula.fit(self.data)

        assert get_qualified_name(
            copula.univariates[0].__class__) == 'copulas.univariate.beta.BetaUnivariate'
        assert get_qualified_name(
            copula.univariates[1].__class__) == 'copulas.univariate.gaussian_kde.GaussianKDE'
        assert get_qualified_name(
            copula.univariates[2].__class__) == 'copulas.univariate.base.Univariate'

    def test_fit_numpy_array(self):
        """Fit should work indistinctly with numpy arrays and pandas dataframes """
        # Setup
        copula = GaussianMultivariate(
            distribution='copulas.univariate.gaussian.GaussianUnivariate')

        # Run
        copula.fit(self.data.values)

        # Check
        for key, (column, univariate) in enumerate(zip(self.data.columns, copula.univariates)):
            assert univariate._params['loc'] == np.mean(self.data[column])
            assert univariate._params['scale'] == np.std(self.data[column])

        expected_covariance = copula._get_covariance(pd.DataFrame(self.data.values))
        assert (copula.covariance == expected_covariance).all().all()

    def test_probability_density(self):
        """Probability_density computes probability for the given values."""
        # Setup
        copula = GaussianMultivariate(GaussianUnivariate)
        copula.fit(self.data)
        X = np.array([2000., 200., 0.])
        expected_result = 0.031163598715950383

        # Run
        result = copula.probability_density(X)

        # Check
        self.assertAlmostEqual(result, expected_result)

    def test_cumulative_distribution_fit_df_call_np_array(self):
        """Cumulative_density integrates the probability density along the given values."""
        # Setup
        copula = GaussianMultivariate(GaussianUnivariate)
        copula.fit(self.data)
        X = np.array([2000., 200., 1.])
        expected_result = 0.4460456536217443

        # Run
        result = copula.cumulative_distribution(X)

        # Check
        assert np.isclose(result, expected_result, atol=1e-5).all().all()

    def test_cumulative_distribution_fit_call_np_array(self):
        """Cumulative_density integrates the probability density along the given values."""
        # Setup
        copula = GaussianMultivariate(GaussianUnivariate)
        copula.fit(self.data.values)
        X = np.array([2000., 200., 1.])
        expected_result = 0.4460456536217443

        # Run
        result = copula.cumulative_distribution(X)

        # Check
        assert np.isclose(result, expected_result, atol=1e-5).all().all()

    def test_cumulative_distribution_fit_call_pd(self):
        """Cumulative_density integrates the probability density along the given values."""
        # Setup
        copula = GaussianMultivariate(GaussianUnivariate)
        copula.fit(self.data.values)
        X = np.array([2000., 200., 1.])
        expected_result = 0.4460456536217443

        # Run
        result = copula.cumulative_distribution(X)

        # Check
        assert np.isclose(result, expected_result, atol=1e-5).all().all()

    @patch('copulas.multivariate.gaussian.np.random.multivariate_normal')
    def test_sample(self, normal_mock):
        """Sample use the inverse-transform method to generate new samples."""
        # Setup
        instance = GaussianMultivariate(GaussianUnivariate)
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
        instance = GaussianMultivariate(GaussianUnivariate, random_seed=0)
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
        pd.testing.assert_frame_equal(result, expected_result, check_less_precise=True)

    def test_to_dict(self):
        """To_dict returns the parameters to replicate the copula."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)

        # Run
        result = copula.to_dict()

        # Asserts
        assert result['type'] == 'copulas.multivariate.gaussian.GaussianMultivariate'
        assert result['columns'] == ['column1', 'column2', 'column3']
        assert len(result['univariates']) == 3

        expected_cov = copula._get_covariance(self.data).tolist()
        np.testing.assert_equal(result['covariance'], expected_cov)

        for univariate, result_univariate in zip(copula.univariates, result['univariates']):
            assert univariate.to_dict() == result_univariate

    def test_from_dict(self):
        """from_dict generates a new instance from its parameters."""
        # Setup
        copula = GaussianMultivariate()
        copula.fit(self.data)
        copula_dict = copula.to_dict()

        # Run
        new_copula = GaussianMultivariate.from_dict(copula_dict)

        # Asserts
        assert isinstance(new_copula, GaussianMultivariate)
        assert new_copula.columns == ['column1', 'column2', 'column3']
        assert len(new_copula.univariates) == 3

        for new_univariate, old_univariate in zip(copula.univariates, new_copula.univariates):
            assert new_univariate.to_dict() == old_univariate.to_dict()

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
