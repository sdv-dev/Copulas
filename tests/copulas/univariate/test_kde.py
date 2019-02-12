#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `kde` module."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy.stats import gaussian_kde

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
        self.kde = KDEUnivariate()
        assert self.kde.model is None

    def test_fit(self):
        """On fit, kde model is instantiated with intance of gaussian_kde."""
        self.kde = KDEUnivariate()
        data = [1, 2, 3, 4, 5]

        self.kde.fit(data)

        self.assertIsInstance(self.kde.model, gaussian_kde)

    def test_fit_uniform(self):
        """On fit, kde model is instantiated with passed data."""
        self.kde = KDEUnivariate()
        data = [1, 2, 3, 4, 5]

        self.kde.fit(data)

        assert self.kde.model

    def test_fit_empty_data(self):
        """If fitting kde model with empty data it will raise ValueError."""
        self.kde = KDEUnivariate()

        with self.assertRaises(ValueError):
            self.kde.fit([])

    def test_probability_density(self):
        """probability_density evaluates with the model."""
        self.setup_norm()

        x = self.kde.probability_density(0.5)

        expected = 0.35206532676429952
        self.assertAlmostEqual(x, expected, places=1)

    def test_cumulative_distribution(self):
        """cumulative_distribution evaluates with the model."""
        self.setup_norm()

        x = self.kde.cumulative_distribution(0.5)

        expected = 0.69146246127401312
        self.assertAlmostEqual(x, expected, places=1)

    def test_percent_point(self):
        """percent_point evaluates with the model."""
        self.setup_norm()

        x = self.kde.percent_point(0.5)

        expected = 0.0
        self.assertAlmostEqual(x, expected, places=1)

    def test_percent_point_invalid_value(self):
        """Evaluating an invalid value will raise ValueError."""
        self.setup_norm()

        with self.assertRaises(ValueError):
            self.kde.percent_point(2)

    @patch('copulas.univariate.kde.scipy.stats.gaussian_kde', autospec=True)
    def test_sample(self, kde_mock):
        """kde.sample is a wrapper calls kde.model.resample."""
        # Setup
        instance = KDEUnivariate()
        X = np.array([1, 2, 3, 4, 5])
        instance.fit(X)

        expected_result = np.array([0, 1, 0, 1, 0])
        model_mock = kde_mock.return_value
        model_mock.resample.return_value = expected_result

        expected_kde_mock_call_args_list = [((np.array([1, 2, 3, 4, 5]),), {})]
        expected_model_mock_call_args_list = [((5,), {})]

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result)
        compare_nested_iterables(kde_mock.call_args_list, expected_kde_mock_call_args_list)
        compare_nested_iterables(model_mock.call_args_list, expected_model_mock_call_args_list)

    def test_sample_random_state(self):
        """If random_state is set, samples will generate the exact same values."""
        # Setup
        instance = KDEUnivariate(random_seed=0)

        X = np.array([1, 2, 3, 4, 5])
        instance.fit(X)

        expected_result_random_state = np.array([
            [7.02156389, 1.45857107, 2.12161148, 7.56801267, 5.14017901]
        ])

        # Run
        result = instance.sample(5)

        # Check
        compare_nested_iterables(result, expected_result_random_state)

    def test_from_dict(self):
        """From_dict sets the values of a dictionary as attributes of the instance."""
        # Setup
        distribution = KDEUnivariate()
        parameters = {
            'd': 1,
            'n': 10,
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
            'covariance': [[0.2081069604419522]],
            'factor': 0.6309573444801932,
            'inv_cov': [[4.805221304834407]]
        }

        # Run
        distribution = KDEUnivariate.from_dict(parameters)

        # Check
        assert distribution.model.d == 1
        assert distribution.model.n == 10
        assert distribution.model.covariance == np.array([[0.2081069604419522]])
        assert distribution.model.factor == 0.6309573444801932
        assert distribution.model.inv_cov == np.array([[4.805221304834407]])
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
            'd': 1,
            'n': 10,
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
            'covariance': [[0.20810696044195218]],
            'factor': 0.6309573444801932,
            'inv_cov': [[4.805221304834407]]
        }

        # Run
        result = distribution.to_dict()

        # Check
        compare_nested_dicts(result, expected_result)
