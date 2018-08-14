#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `copulas` package."""

from unittest import TestCase

import numpy as np

from copulas.bivariate.base import Bivariate, CopulaTypes


class TestCopulas(TestCase):
    def setUp(self):

        self.U = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.V = [0.5, 0.6, 0.8, 0.7, 0.6]

        self.c0 = Bivariate(CopulaTypes.CLAYTON)
        self.c0.fit(self.U, self.V)
        self.c1 = Bivariate('Frank')
        self.c1.fit(self.U, self.V)
        self.c2 = Bivariate('Gumbel')
        self.c2.fit(self.U, self.V)

    def test_snippet(self):

        U = [0.1, 0.2, 0.3, 0.4]
        V = [0.5, 0.6, 0.5, 0.8]

        c0 = Bivariate('Clayton')
        c0.fit(U, V)
        result = c0.cumulative_density([0, 0.1, 0.2], [0, 0.1, 0.8])
        expected_result = [0, 0.07517146687679954, 0.19881186077542212]
        assert (result == expected_result).all()

        c2 = Bivariate('Gumbel')
        c2.fit(U, V)
        result = c2.cumulative_density([0, 0.1, 0.2], [0, 0.1, 0.8])
        expected_result = np.array([0.0, 0.042835279521916785, 0.19817042125347709])
        for i in range(3):
            assert result[i] == expected_result[i]
        assert (result == expected_result).all()

        # LOGGER.debug(Copula.select_copula(U,V))

    def test_fit(self):
        """cross-check fit with matlab implementation
        """
        result = self.c0.theta
        expected_result = 0.9250
        self.assertAlmostEquals(result, expected_result, places=3)

        result = self.c1.theta
        expected_result = 3.1037
        self.assertAlmostEquals(result, expected_result, places=3)

        result = self.c2.theta
        expected_result = 1.4625
        self.assertAlmostEquals(result, expected_result, places=3)

    def test_pdf(self):

        result = self.c0.probability_density(0.1, 0.5)
        expected_result = 0.6355
        self.assertAlmostEquals(result, expected_result, places=3)

        result = self.c1.probability_density(0.1, 0.5)
        expected_result = 0.830
        self.assertAlmostEquals(result, expected_result, places=3)

        result = self.c2.probability_density(0.1, 0.5)
        expected_result = 0.9395
        self.assertAlmostEquals(result, expected_result, places=3)

    def test_cdf(self):

        result = self.c0.cumulative_density(0.1, 0.5)
        expected_result = 0.0896
        self.assertAlmostEquals(result, expected_result, places=3)

        result = self.c1.cumulative_density(0.1, 0.5)
        expected_result = 0.0801
        self.assertAlmostEquals(result, expected_result, places=3)

        result = self.c2.cumulative_density(0.1, 0.5)
        expected_result = 0.0767
        self.assertAlmostEquals(result, expected_result, places=3)

    def test_frank_cdf_invalid_value(self):

        result = self.c1.cumulative_density(0.2, 0.3)
        expected_result = 0.1119
        self.assertAlmostEquals(result, expected_result, places=3)

    def test_sample(self):
        result = self.c0.sample(10)
        assert result.shape[0] == 10

        result = self.c1.sample(10)
        assert result.shape[0] == 10

        result = self.c2.sample(10)
        assert result.shape[0] == 10

    def test_copula_selction_negative_tau(self):
        """If tau is negative, should choose frank copula."""
        U = [0.1, 0.2, 0.3, 0.4]
        V = [0.6, 0.5, 0.4, 0.3]

        name, param = Bivariate.select_copula(U, V)
        expected = CopulaTypes.FRANK

        assert name == expected

    def test_copula_selction(self):
        U = [0.1, 0.2, 0.3, 0.4]
        V = [1, 2, 3, 4]

        name, param = Bivariate.select_copula(U, V)
        expected = CopulaTypes.CLAYTON.value

        assert name == expected
