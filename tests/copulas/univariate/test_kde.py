#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `univariate.kde` module."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from copulas.univariate.kde import KDEUnivariate
from tests import compare_nested_dicts, compare_nested_iterables


class TestKDEUnivariate(TestCase):

    def setup_norm(self):
        """set up the model to fit standard norm data."""
        self.kde = KDEUnivariate()
        # use 42 as a fixed random seed
        np.random.seed(42)
        column = np.random.normal(0, 1, 1000)
        self.kde.fit(column)

    def test___init__(self):
        """On init, model are set to None."""
        # Setup / Run
        instance = KDEUnivariate()

        # Check
        instance.model is None
        instance.fitted is False
        instance.constant_value is None
        instance.sample_size == 10

    @patch('copulas.univariate.kde.scipy.stats.gaussian_kde', autospec=True)
    def test_fit(self, kde_mock):
        """On fit, a new instance of gaussian_kde is fitted."""
        # Setup
        instance = KDEUnivariate()
        X = np.array([1, 2, 3, 4, 5])

        kde_instance_mock = kde_mock.return_value

        # Run
        instance.fit(X)

        # Check
        assert instance.model == kde_instance_mock
        assert instance.fitted is True
        assert instance.constant_value is None
        kde_mock.assert_called_once_with(X)

    def test_fit_constant(self):
        """If fit data is constant, no gaussian_kde model is created."""
        # Setup
        instance = KDEUnivariate()
        X = np.array([1, 1, 1, 1, 1])

        # Run
        instance.fit(X)

        # Check
        assert instance.model is None
        assert instance.constant_value == 1
        assert instance.fitted is True

    @patch('copulas.univariate.kde.scipy.stats.gaussian_kde', autospec=True)
    def test_probability_density(self, kde_mock):
        """Probability_density evaluates with the model."""
        # Setup
        kde_instance = kde_mock.return_value
        kde_instance.evaluate.return_value = ('probability_density_from_model',)

        instance = KDEUnivariate()
        X = np.array([1, 2, 3, 4, 5])

        instance.fit(X)

        # Run
        result = instance.probability_density(0.5)

        assert result == 'probability_density_from_model'
        kde_instance.evaluate.assert_called_once_with(0.5)

    def test_cumulative_distribution(self):
        """cumulative_distribution evaluates with the model."""
        self.setup_norm()

        x = self.kde.cumulative_distribution(0.5)

        expected = 0.69146246127401312
        self.assertAlmostEqual(x, expected, places=1)

    def test_percent_point(self):
        """percent_point evaluates with the model."""
        instance = KDEUnivariate(sample_size=1000)

        np.random.seed(42)
        X = np.random.normal(0, 1, 1000)

        instance.fit(X)

        expected_result = 0.0

        # Run
        result = instance.percent_point(0.5)

        # Check
        np.testing.assert_allclose(result, expected_result, atol=1e-1)
        # TODO: Discuss if this loss of precision is acceptable

    def test_percent_point_invalid_value(self):
        """Evaluating an invalid value will raise ValueError."""
        self.setup_norm()

        with self.assertRaises(ValueError):
            self.kde.percent_point(2)

    @patch('copulas.univariate.kde.scipy.stats.gaussian_kde', autospec=True)
    def test_sample(self, kde_mock):
        """When fitted, we are able to use the model to get samples."""
        # Setup
        model_mock = kde_mock.return_value
        model_mock.resample.return_value = np.array([0, 1, 0, 1, 0])

        instance = KDEUnivariate()
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
        instance = KDEUnivariate()
        instance.fitted = True
        instance.constant_value = 3
        instance._replace_constant_methods()

        expected_result = np.array([3, 3, 3, 3, 3])

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)

    def test_sample_random_state(self):
        """If random_state is set, samples will generate the exact same values."""
        # Setup
        instance = KDEUnivariate(random_seed=0)

        X = np.array([1, 2, 3, 4, 5])
        instance.fit(X)

        expected_result_random_state = np.array([
            [5.02156389, 5.45857107, 6.12161148, 4.56801267, 6.14017901]
        ])

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result_random_state)

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
        distribution = KDEUnivariate.from_dict(parameters)

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

    def test_to_dict(self):
        """To_dict returns the defining parameters of a distribution in a dict."""
        # Setup
        distribution = KDEUnivariate()
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
        distribution.fit(column)

        expected_result = {
            'type': 'copulas.univariate.kde.KDEUnivariate',
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
        instance = KDEUnivariate()

        # Run
        result = KDEUnivariate.from_dict(instance.to_dict())

        # Check
        assert instance.to_dict() == result.to_dict()

    def test_valid_serialization_fit_model(self):
        """For a fitted model to_dict and from_dict are opposites."""
        # Setup
        instance = KDEUnivariate()
        X = np.array([1, 2, 3, 4])
        instance.fit(X)

        # Run
        result = KDEUnivariate.from_dict(instance.to_dict())

        # Check
        assert instance.to_dict() == result.to_dict()

    @patch('copulas.univariate.base.Univariate._constant_probability_density', autospec=True)
    def test_probability_density_constant(self, pdf_mock):
        """If constant_value, probability_density uses the degenerate version."""
        # Setup
        instance = KDEUnivariate()
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
        instance = KDEUnivariate()
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
