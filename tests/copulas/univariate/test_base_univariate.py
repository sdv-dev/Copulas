from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from copulas.univariate.base import ScipyWrapper, Univariate
from tests import compare_nested_iterables


class TestUnivariate(TestCase):

    def test_get_constant_value(self):
        """get_constant_value return the unique value of an array if it exists."""
        # Setup
        X = np.array([1, 1, 1, 1])
        expected_result = 1

        # Run
        result = Univariate._get_constant_value(X)

        # Check
        assert result == expected_result

    def test_get_constant_value_non_constant(self):
        """get_constant_value return None on non-constant arrays."""
        # Setup
        X = np.array(range(5))
        expected_result = None

        # Run
        result = Univariate._get_constant_value(X)

        # Check
        assert result is expected_result

    def test__sample_sample(self):
        """_constant_sample returns a constant array of num_samples length."""
        # Setup
        instance = Univariate()
        instance.constant_value = 15

        expected_result = np.array([15, 15, 15, 15, 15])

        # Run
        result = instance._constant_sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

    def test__constant_cumulative_distribution(self):
        """constant_cumulative_distribution returns only 0 and 1."""
        # Setup
        instance = Univariate()
        instance.constant_value = 3

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
        instance.constant_value = 3

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
        instance.constant_value = 3

        X = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_result = np.array([3, 3, 3, 3, 3, 3])

        # Run
        result = instance._constant_percent_point(X)

        # Check
        compare_nested_iterables(result, expected_result)


class TestScipyWrapper(TestCase):

    def test___init___valid_params(self):
        """On init, self.model is set to None."""
        # Setup
        class ScipyWrapperSubclass(ScipyWrapper):
            model_class = 'gaussian_kde'

        # Run
        instance = ScipyWrapperSubclass()

        # Check
        assert instance.model is None
        assert instance.fitted is False
        assert instance.constant_value is None

    def test_fit_constant(self):
        # Run
        wrapper = ScipyWrapper()
        wrapper.fit(np.zeros(5))

        # Asserts
        assert wrapper.fitted
        assert wrapper.cumulative_distribution == wrapper._constant_cumulative_distribution
        assert wrapper.percent_point == wrapper._constant_percent_point
        assert wrapper.probability_density == wrapper._constant_probability_density
        assert wrapper.sample == wrapper._constant_sample

    @patch('copulas.univariate.base.scipy.stats')
    def test_fit_fittable(self, stats_mock):
        # Setup
        class ScipyWrapperSubclass(ScipyWrapper):
            model_class = 'dummy'
            cumulative_distribution = 'cdf'
            percent_point = 'ppf'
            probability_density = 'pdf'
            sample = 'rvs'

        dummy = MagicMock()
        stats_mock.dummy.return_value = dummy

        # Run
        data = np.array(range(5))
        wrapper = ScipyWrapperSubclass()
        wrapper.fit(data, 'some', 'args', some='kwargs')

        # Asserts
        stats_mock.dummy.assert_called_once_with(data, 'some', 'args', some='kwargs')
        assert wrapper.model == dummy

        assert wrapper.fitted
        assert wrapper.cumulative_distribution == dummy.cdf
        assert wrapper.percent_point == dummy.ppf
        assert wrapper.probability_density == dummy.pdf
        assert wrapper.sample == dummy.rvs

    @patch('copulas.univariate.base.scipy.stats')
    def test_fit_unfittable(self, stats_mock):
        # Setup
        class ScipyWrapperSubclass(ScipyWrapper):
            unfittable_model = True
            model_class = 'dummy'
            cumulative_distribution = 'cdf'
            percent_point = 'ppf'
            probability_density = 'pdf'
            sample = 'rvs'

        dummy = MagicMock()
        stats_mock.dummy.return_value = dummy

        # Run
        data = np.array(range(5))
        wrapper = ScipyWrapperSubclass()
        wrapper.fit(data, 'some', 'args', some='kwargs')

        # Asserts
        stats_mock.dummy.assert_called_once_with('some', 'args', some='kwargs')
        assert wrapper.model == dummy

        assert wrapper.fitted
        assert wrapper.cumulative_distribution == dummy.cdf
        assert wrapper.percent_point == dummy.ppf
        assert wrapper.probability_density == dummy.pdf
        assert wrapper.sample == dummy.rvs

    @patch('copulas.univariate.base.scipy.stats', autospec=True)
    def test_probability_density(self, scipy_mock):
        """probability_density calls to the mapped method of model."""
        # Setup
        class ScipyWrapperSubclass(ScipyWrapper):
            model_class = 'mock_model'
            probability_density = 'pdf'
            cumulative_distribution = None
            percent_point = None
            sample = None

        model_instance_mock = MagicMock(spec=['pdf'])
        model_instance_mock.pdf.return_value = 'pdf value'
        model_class_mock = MagicMock()
        model_class_mock.return_value = model_instance_mock
        scipy_mock.mock_model = model_class_mock

        fit_data = np.array(range(5))
        instance = ScipyWrapperSubclass()
        instance.fit(fit_data)

        call_data = np.array([0.0])
        expected_result = 'pdf value'

        # Run
        result = instance.probability_density(call_data)

        # Check
        assert result == expected_result

        scipy_mock.assert_not_called()
        model_class_mock.assert_called_once_with(fit_data)
        model_instance_mock.assert_not_called()
        model_instance_mock.pdf.assert_called_once_with(call_data)

    @patch('copulas.univariate.base.scipy.stats', autospec=True)
    def test_cumulative_distribution(self, scipy_mock):
        """cumulative_distribution calls to the mapped method of model."""
        # Setup
        class ScipyWrapperSubclass(ScipyWrapper):
            model_class = 'mock_model'
            probability_density = None
            cumulative_distribution = 'cdf'
            percent_point = None
            sample = None

        model_instance_mock = MagicMock(spec=['cdf'])
        model_instance_mock.cdf.return_value = 'cdf value'
        model_class_mock = MagicMock()
        model_class_mock.return_value = model_instance_mock
        scipy_mock.mock_model = model_class_mock

        fit_data = np.array(range(5))
        instance = ScipyWrapperSubclass()
        instance.fit(fit_data)

        call_data = np.array([0.0])
        expected_result = 'cdf value'

        # Run
        result = instance.cumulative_distribution(call_data)

        # Check
        assert result == expected_result

        scipy_mock.assert_not_called()
        model_class_mock.assert_called_once_with(fit_data)
        model_instance_mock.assert_not_called()
        model_instance_mock.cdf.assert_called_once_with(call_data)

    @patch('copulas.univariate.base.scipy.stats', autospec=True)
    def test_percent_point(self, scipy_mock):
        """percent_point calls to the mapped method of model."""
        # Setup
        class ScipyWrapperSubclass(ScipyWrapper):
            model_class = 'mock_model'
            probability_density = None
            cumulative_distribution = None
            percent_point = 'ppf'
            sample = None

        model_instance_mock = MagicMock(spec=['ppf'])
        model_instance_mock.ppf.return_value = 'ppf value'
        model_class_mock = MagicMock()
        model_class_mock.return_value = model_instance_mock
        scipy_mock.mock_model = model_class_mock

        fit_data = np.array(range(5))
        instance = ScipyWrapperSubclass()
        instance.fit(fit_data)

        call_data = np.array([0.0])
        expected_result = 'ppf value'

        # Run
        result = instance.percent_point(call_data)

        # Check
        assert result == expected_result

        scipy_mock.assert_not_called()
        model_class_mock.assert_called_once_with(fit_data)
        model_instance_mock.assert_not_called()
        model_instance_mock.ppf.assert_called_once_with(call_data)

    @patch('copulas.univariate.base.scipy.stats', autospec=True)
    def test_sample(self, scipy_mock):
        """sample calls to the mapped method of model."""
        # Setup
        class ScipyWrapperSubclass(ScipyWrapper):
            model_class = 'mock_model'
            probability_density = None
            cumulative_distribution = None
            percent_point = None
            sample = 'sample'

        model_instance_mock = MagicMock(spec=['sample'])
        model_instance_mock.sample.return_value = 'samples'
        model_class_mock = MagicMock()
        model_class_mock.return_value = model_instance_mock
        scipy_mock.mock_model = model_class_mock

        fit_data = np.array(range(5))
        instance = ScipyWrapperSubclass()
        instance.fit(fit_data)

        call_data = np.array([0.0])
        expected_result = 'samples'

        # Run
        result = instance.sample(call_data)

        # Check
        assert result == expected_result

        scipy_mock.assert_not_called()
        model_class_mock.assert_called_once_with(fit_data)
        model_instance_mock.assert_not_called()
        model_instance_mock.sample.assert_called_once_with(call_data)

    def test_from_dict(self):
        """_from_dict will raise NotImpementedError."""
        # Run / Check
        with self.assertRaises(NotImplementedError):
            ScipyWrapper.from_dict({})
