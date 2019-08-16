#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `univariate.kde` module."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from copulas.univariate.gaussian_kde import GaussianKDE
from tests import compare_nested_dicts, compare_nested_iterables


class TestGaussianKDE(TestCase):

    def setup_norm(self):
        """set up the model to fit standard norm data."""
        self.kde = GaussianKDE()
        # use 42 as a fixed random seed
        np.random.seed(42)
        column = np.random.normal(0, 1, 1000)
        self.kde.fit(column)

    def test___init__(self):
        """On init, model are set to None."""
        # Setup / Run
        instance = GaussianKDE()

        # Check
        instance.model is None
        instance.fitted is False
        instance.constant_value is None

    @patch('copulas.univariate.gaussian_kde.scipy.stats.gaussian_kde', autospec=True)
    def test_fit(self, kde_mock):
        """On fit, a new instance of gaussian_kde is fitted."""
        # Setup
        instance = GaussianKDE()
        X = np.array([1, 2, 3, 4, 5])

        def sample_method(*args, **kwargs):
            return X

        kde_instance = MagicMock(evaluate='pdf', resample=sample_method)
        kde_mock.return_value = kde_instance

        # Run
        instance.fit(X)

        # Check
        assert instance.model == kde_instance
        assert instance.fitted is True
        assert instance.constant_value is None
        assert instance.sample == sample_method
        assert instance.probability_density == 'pdf'
        kde_mock.assert_called_once_with(X)

    def test_fit_constant(self):
        """If fit data is constant, no gaussian_kde model is created."""
        # Setup
        instance = GaussianKDE()
        X = np.array([1, 1, 1, 1, 1])

        # Run
        instance.fit(X)

        # Check
        assert instance.model is None
        assert instance.constant_value == 1
        assert instance.fitted is True

    def test_fit_empty_data(self):
        """If fitting kde model with empty data it will raise ValueError."""
        # Setup
        instance = GaussianKDE()
        data = np.array([])

        # Run / Check
        with self.assertRaises(ValueError):
            instance.fit(data)

    @patch('copulas.univariate.gaussian_kde.scalarize', autospec=True)
    @patch('copulas.univariate.gaussian_kde.partial', autospec=True)
    def test__brentq_cdf(self, partial_mock, scalarize_mock):
        """_brentq_cdf returns a function that computes the cdf of a scalar minus its argument."""
        # Setup
        instance = GaussianKDE()

        def mock_partial_return_value(x):
            return x

        scalarize_mock.return_value = 'scalar_function'
        partial_mock.return_value = mock_partial_return_value

        # Run
        result = instance._brentq_cdf(0.5)

        # Check
        assert callable(result)

        # result uses the return_value of partial_mock, so every value returned
        # is (x - 0.5)
        assert result(1.0) == 0.5
        assert result(0.5) == 0
        assert result(0.0) == -0.5

        scalarize_mock.assert_called_once_with(GaussianKDE.cumulative_distribution)
        partial_mock.assert_called_once_with('scalar_function', instance)

    @patch('copulas.univariate.gaussian_kde.scipy.stats.gaussian_kde', autospec=True)
    def test_probability_density(self, kde_mock):
        """probability_density evaluates with the model."""
        # Setup
        fit_data = np.array([1, 2, 3, 4, 5])
        model_mock = kde_mock.return_value
        model_mock.evaluate.return_value = np.array([0.0, 0.5, 1.0])
        model_mock.resample.return_value = fit_data

        instance = GaussianKDE()
        instance.fit(fit_data)
        call_data = np.array([-10, 0, 10])

        expected_result = np.array([0.0, 0.5, 1.0])

        # Run
        result = instance.probability_density(call_data)

        # Check
        compare_nested_iterables(result, expected_result)

        # <kde_mock.assert_called_once_with(fit_data)
        model_mock.evaluate.assert_called_once_with(call_data)

    @patch('copulas.univariate.gaussian_kde.scipy.stats.gaussian_kde', autospec=True)
    def test_cumulative_distribution(self, kde_mock):
        """cumulative_distribution evaluates with the model."""
        # Setup
        model_mock = kde_mock.return_value
        model_mock.integrate_box_1d.side_effect = [0.0, 0.5, 1.0]

        model_mock.dataset = MagicMock()
        model_mock.dataset.mean.return_value = 1
        model_mock.dataset.std.return_value = 0.1

        fit_data = np.array([1, 2, 3, 4, 5])
        instance = GaussianKDE()
        instance.fit(fit_data)

        call_data = np.array([-10, 0, 10])
        expected_result = np.array([0.0, 0.5, 1.0])

        expected_integrate_1d_box_call_args_list = [
            ((0.5, -10), {}),  # The first argument is the lower_bound (1 - 0.1*5)
            ((0.5, 0), {}),
            ((0.5, 10), {}),
        ]

        # Run
        result = instance.cumulative_distribution(call_data)

        # Check
        compare_nested_iterables(result, expected_result)

        kde_mock.assert_called_once_with(fit_data)
        assert (model_mock.integrate_box_1d.call_args_list
                == expected_integrate_1d_box_call_args_list)

    @patch('copulas.univariate.gaussian_kde.GaussianKDE._brentq_cdf', autospec=True)
    @patch('copulas.univariate.gaussian_kde.scipy.optimize.brentq', autospec=True)
    @patch('copulas.univariate.gaussian_kde.scipy.stats.gaussian_kde', autospec=True)
    def test_percent_point(self, kde_mock, brentq_mock, cdf_mock):
        """percent_point evaluates with the model."""
        # Setup
        model_mock = kde_mock.return_value
        brentq_mock.return_value = -250.0
        cdf_mock.return_value = 'a nice scalar bounded method'

        fit_data = np.array([1, 2, 3, 4, 5])
        instance = GaussianKDE()
        instance.fit(fit_data)

        expected_result = np.array([-250.0])

        # Run
        result = instance.percent_point([0.5])

        # Check
        assert result == expected_result

        kde_mock.assert_called_once_with(fit_data)
        model_mock.assert_not_called()
        assert len(model_mock.method_calls) == 0

        brentq_mock.assert_called_once_with('a nice scalar bounded method', -1000, 1000)

    def test_percent_point_invalid_value(self):
        """Evaluating an invalid value will raise ValueError."""
        fit_data = np.array([1, 2, 3, 4, 5])
        instance = GaussianKDE()
        instance.fit(fit_data)

        with self.assertRaises(ValueError):
            instance.percent_point([2.])

    def test_from_dict(self):
        """From_dict sets the values of a dictionary as attributes of the instance."""
        # Setup
        parameters = {
            'fitted': True,
            'dataset': [[
                0.4967141530112327,
                -0.13826430117118466,
                0.6476885381006925,
                1.5230298564080254,
                -0.23415337472333597,
                -0.23413695694918055,
                1.5792128155073915,
                0.7674347291529088,
                -0.4694743859349521,
                0.5425600435859647
            ]],
        }

        # Run
        distribution = GaussianKDE.from_dict(parameters)

        # Check
        assert distribution.model.d == 1
        assert distribution.model.n == 10
        assert distribution.model.covariance == np.array([[0.20810696044195226]])
        assert distribution.model.factor == 0.6309573444801932
        assert distribution.model.inv_cov == np.array([[4.805221304834406]])
        assert (distribution.model.dataset == np.array([[
            0.4967141530112327,
            -0.13826430117118466,
            0.6476885381006925,
            1.5230298564080254,
            -0.23415337472333597,
            -0.23413695694918055,
            1.5792128155073915,
            0.7674347291529088,
            -0.4694743859349521,
            0.5425600435859647
        ]])).all()

    @patch('copulas.univariate.kde.scipy.stats.gaussian_kde', autospec=True)
    def test_to_dict(self, kde_mock):
        """To_dict returns the defining parameters of a distribution in a dict."""
        # Setup
        column = np.array([[
            0.4967141530112327,
            -0.13826430117118466,
            0.6476885381006925,
            1.5230298564080254,
            -0.23415337472333597,
            -0.23413695694918055,
            1.5792128155073915,
            0.7674347291529088,
            -0.4694743859349521,
            0.5425600435859647
        ]])

        kde_instance_mock = kde_mock.return_value
        kde_instance_mock.dataset = column
        kde_instance_mock.resample.return_value = column
        distribution = GaussianKDE()
        distribution.fit(column)

        expected_result = {
            'type': 'copulas.univariate.gaussian_kde.GaussianKDE',
            'fitted': True,
            'dataset': [[
                0.4967141530112327,
                -0.13826430117118466,
                0.6476885381006925,
                1.5230298564080254,
                -0.23415337472333597,
                -0.23413695694918055,
                1.5792128155073915,
                0.7674347291529088,
                -0.4694743859349521,
                0.5425600435859647
            ]],
        }

        # Run
        result = distribution.to_dict()

        # Check
        compare_nested_dicts(result, expected_result)

    def test_valid_serialization_unfit_model(self):
        """For a unfitted model to_dict and from_dict are opposites."""
        # Setup
        instance = GaussianKDE()

        # Run
        result = GaussianKDE.from_dict(instance.to_dict())

        # Check
        assert instance.to_dict() == result.to_dict()

    def test_valid_serialization_fit_model(self):
        """For a fitted model to_dict and from_dict are opposites."""
        # Setup
        instance = GaussianKDE()
        X = np.array([1, 2, 3, 4])
        instance.fit(X)

        # Run
        result = GaussianKDE.from_dict(instance.to_dict())

        # Check
        assert instance.to_dict() == result.to_dict()

    @patch('copulas.univariate.kde.scipy.stats.gaussian_kde', autospec=True)
    def test_sample(self, kde_mock):
        """When fitted, we are able to use the model to get samples."""
        # Setup
        model_mock = kde_mock.return_value
        model_mock.resample.return_value = np.array([0, 1, 0, 1, 0])

        instance = GaussianKDE()
        X = np.array([1, 2, 3, 4, 5])
        instance.fit(X)

        expected_result = np.array([0, 1, 0, 1, 0])

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

        assert instance.model == model_mock
        kde_mock.assert_called_once_with(X)
        model_mock.resample.assert_called_once_with(5)

    def test_sample_constant(self):
        """If constant_value is set, all the sample have the same value."""
        # Setup
        instance = GaussianKDE()
        instance.fitted = True
        instance.constant_value = 3
        instance._replace_constant_methods()

        expected_result = np.array([3, 3, 3, 3, 3])

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

    @patch('copulas.univariate.base.Univariate._constant_probability_density', autospec=True)
    def test_probability_density_constant(self, pdf_mock):
        """If constant_value, probability_density uses the degenerate version."""
        # Setup
        instance = GaussianKDE()
        instance.fitted = True
        instance.constant_value = 3
        instance._replace_constant_methods()

        X = np.array([0, 1, 2, 3, 4, 5])
        expected_result = np.array([0, 0, 1, 0, 0])

        pdf_mock.return_value = np.array([0, 0, 1, 0, 0])

        # Run
        result = instance.probability_density(X)

        # Check
        compare_nested_iterables(result, expected_result)
        pdf_mock.assert_called_once_with(instance, X)

    @patch('copulas.univariate.base.Univariate._constant_percent_point', autospec=True)
    def test_percent_point_constant_raises(self, ppf_mock):
        """If constant_value, percent_point uses the degenerate version."""
        # Setup
        instance = GaussianKDE()
        instance.fitted = True
        instance.constant_value = 3
        instance._replace_constant_methods()

        X = np.array([0.1, 0.5, 0.75])
        expected_result = np.array([3, 3, 3])

        ppf_mock.return_value = np.array([3, 3, 3])

        # Run
        result = instance.percent_point(X)

        # Check
        compare_nested_iterables(result, expected_result)
        ppf_mock.assert_called_once_with(instance, X)
