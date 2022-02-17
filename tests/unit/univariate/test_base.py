from unittest.mock import patch

import numpy as np

from copulas.univariate.base import BoundedType, ParametricType, Univariate
from copulas.univariate.beta import BetaUnivariate
from copulas.univariate.gamma import GammaUnivariate
from copulas.univariate.gaussian import GaussianUnivariate
from copulas.univariate.gaussian_kde import GaussianKDE
from copulas.univariate.log_laplace import LogLaplace
from copulas.univariate.student_t import StudentTUnivariate
from copulas.univariate.truncated_gaussian import TruncatedGaussian
from copulas.univariate.uniform import UniformUnivariate
from tests import compare_nested_iterables


class TestUnivariate:

    def test__select_candidates(self):
        # Run
        candidates = Univariate._select_candidates()

        # Assert
        assert set(candidates) == {
            GaussianKDE,
            GaussianUnivariate,
            TruncatedGaussian,
            BetaUnivariate,
            GammaUnivariate,
            StudentTUnivariate,
            UniformUnivariate,
            LogLaplace
        }

    def test__select_candidates_parametric(self):
        # Run
        candidates = Univariate._select_candidates(parametric=ParametricType.PARAMETRIC)

        # Assert
        assert set(candidates) == {
            GaussianUnivariate,
            TruncatedGaussian,
            BetaUnivariate,
            GammaUnivariate,
            StudentTUnivariate,
            UniformUnivariate,
            LogLaplace
        }

    def test__select_candidates_non_parametric(self):
        # Run
        candidates = Univariate._select_candidates(parametric=ParametricType.NON_PARAMETRIC)

        # Assert
        assert candidates == [GaussianKDE]

    def test__select_candidates_bounded(self):
        # Run
        candidates = Univariate._select_candidates(bounded=BoundedType.BOUNDED)

        # Assert
        assert set(candidates) == {
            TruncatedGaussian,
            BetaUnivariate,
            UniformUnivariate
        }

    def test__select_candidates_unbounded(self):
        # Run
        candidates = Univariate._select_candidates(bounded=BoundedType.UNBOUNDED)

        # Assert
        assert set(candidates) == {
            GaussianKDE,
            GaussianUnivariate,
            StudentTUnivariate
        }

    def test__select_candidates_semibounded(self):
        # Run
        candidates = Univariate._select_candidates(bounded=BoundedType.SEMI_BOUNDED)

        # Assert
        assert set(candidates) == {
            GammaUnivariate,
            LogLaplace
        }

    def test_fit_constant(self):
        """If constant values, replace methods."""
        # Setup
        distribution = Univariate()

        # Run
        distribution.fit(np.array([1, 1, 1, 1, 1]))

        # Assert
        assert distribution.fitted
        assert distribution._instance._is_constant()

    def test_fit_not_constant(self):
        """If constant values, replace methods."""
        # Setup
        distribution = Univariate()

        # Run
        distribution.fit(np.array([1, 2, 3, 4, 1]))

        # Assert
        assert distribution.fitted
        assert not distribution._instance._is_constant()

    @patch('copulas.univariate.base.select_univariate')
    def test_fit_selection_sample_size_small(self, select_mock):
        """if selection_sample_size is smaller than data, subsample the data before selecting."""
        # Setup
        distribution = Univariate(selection_sample_size=3)

        # Run
        distribution.fit(np.array([1, 1, 1, 1, 1]))

        # Assert
        assert distribution.fitted
        assert distribution._instance == select_mock.return_value

        call_args = select_mock.call_args_list
        selection_sample = call_args[0][0][0]
        np.testing.assert_array_equal(selection_sample, np.array([1, 1, 1]))

        fit_call_args = select_mock.return_value.fit.call_args_list
        np.testing.assert_array_equal(fit_call_args[0][0][0], np.array([1, 1, 1, 1, 1]))

    @patch('copulas.univariate.base.select_univariate')
    def test_fit_selection_sample_size_large(self, select_mock):
        """if selection_sample_size is smaller than data, subsample the data before selecting."""
        # Setup
        distribution = Univariate(selection_sample_size=10)

        # Run
        distribution.fit(np.array([1, 1, 1, 1, 1]))

        # Assert
        assert distribution.fitted
        assert distribution._instance == select_mock.return_value

        call_args = select_mock.call_args_list
        selection_sample = call_args[0][0][0]
        np.testing.assert_array_equal(selection_sample, np.array([1, 1, 1, 1, 1]))

        fit_call_args = select_mock.return_value.fit.call_args_list
        np.testing.assert_array_equal(fit_call_args[0][0][0], np.array([1, 1, 1, 1, 1]))

    def test_check_constant_value(self):
        """check_constant_value return True if the array is constant."""
        # Setup
        X = np.array([1, 1, 1, 1])

        # Run
        uni = Univariate()
        constant = uni._check_constant_value(X)

        # Check
        assert constant

    def test_check_constant_value_non_constant(self):
        """_check_constant_value returns False if the array is not constant."""
        # Setup
        X = np.array([1, 2, 3, 4])

        # Run
        uni = Univariate()
        constant = uni._check_constant_value(X)

        # Check
        assert not constant

    def test__constant_sample(self):
        """_constant_sample returns a constant array of num_samples length."""
        # Setup
        instance = Univariate()
        instance._constant_value = 15

        expected_result = np.array([15, 15, 15, 15, 15])

        # Run
        result = instance._constant_sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

    def test__constant_cumulative_distribution(self):
        """constant_cumulative_distribution returns only 0 and 1."""
        # Setup
        instance = Univariate()
        instance._constant_value = 3

        X = np.array([1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 1, 1, 1])

        # Run
        result = instance._constant_cumulative_distribution(X)

        # Check
        compare_nested_iterables(result, expected_result)

    def test__constant_probability_density(self):
        """constant_probability_density only is 1 in self.constant_value."""
        # Setup
        instance = Univariate()
        instance._constant_value = 3

        X = np.array([1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 1, 0, 0])

        # Run
        result = instance._constant_probability_density(X)

        # Check
        compare_nested_iterables(result, expected_result)

    def test__constant_percent_point(self):
        """constant_percent_point only is self.constant_value in non-zero probabilities."""
        # Setup
        instance = Univariate()
        instance._constant_value = 3

        X = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_result = np.array([3, 3, 3, 3, 3, 3])

        # Run
        result = instance._constant_percent_point(X)

        # Check
        compare_nested_iterables(result, expected_result)

    def test_set_random_state(self):
        """Test `set_random_state` works as expected."""
        # Setup
        instance = Univariate()

        # Run
        instance.set_random_state(3)

        # Check
        assert isinstance(instance.random_state, np.random.RandomState)
