#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `univariate` package."""

from unittest import TestCase

import numpy as np
import scipy

from copulas.univariate.kde import KDEUnivariate
from tests import compare_nested_dicts


class TestKDEUnivariate(TestCase):
    def setup_norm(self):
        """set up the model to fit standard norm data"""
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
        """On fit, kde model is instantiated with intance of gaussian_kde"""
        self.kde = KDEUnivariate()
        data = [1, 2, 3, 4, 5]

        self.kde.fit(data)

        self.assertIsInstance(self.kde.model, scipy.stats.gaussian_kde)

    def test_fit_uniform(self):
        """On fit, kde model is instantiated with passed data"""
        self.kde = KDEUnivariate()
        data = [1, 2, 3, 4, 5]

        self.kde.fit(data)

        assert self.kde.data == data

    def test_fit_empty_data(self):
        """If fitting kde model with empty data it will raise ValueError"""
        self.kde = KDEUnivariate()

        with self.assertRaises(ValueError):
            self.kde.fit([])

    def test_get_pdf(self):
        """get_pdf evaluates with the model"""
        self.setup_norm()

        x = self.kde.get_pdf(0.5)

        expected = 0.35206532676429952
        self.assertAlmostEquals(x, expected, places=1)

    def test_get_cdf(self):
        """get_pdf evaluates with the model"""
        self.setup_norm()

        x = self.kde.get_cdf(0.5)

        expected = 0.69146246127401312
        self.assertAlmostEquals(x, expected, places=1)

    def test_get_ppf(self):
        """get_ppf evaluates with the model"""
        self.setup_norm()

        x = self.kde.get_ppf(0.5)

        expected = 0.0
        self.assertAlmostEquals(x, expected, places=1)

    def test_get_ppf_invalid_value(self):
        """Evaluating an invalid value will raise ValueError"""
        self.setup_norm()

        with self.assertRaises(ValueError):
            self.kde.get_ppf(2)

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
