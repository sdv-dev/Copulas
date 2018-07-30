#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `univariate` package."""

from unittest import TestCase

import numpy as np

from copulas.univariate.KDEUnivariate import KDEUnivariate


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

    def test_fit_uniform(self):
        """On fit, kde model is instantiated with passed data"""
        self.kde = KDEUnivariate()
        self.kde.fit([1, 2, 3, 4, 5])
        assert self.kde.model is not None

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

    def test_get_pdf_invalid_value(self):
        """Evaluating an invalid value will crash"""

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
