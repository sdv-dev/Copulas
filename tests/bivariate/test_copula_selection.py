#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `copulas` package."""

from unittest import TestCase

import numpy as np

from copulas.bivariate.copulas import Copula


class TestCopulas(TestCase):

    def test_snippet(self):

        U = [0.1, 0.2, 0.3, 0.4]
        V = [0.5, 0.6, 0.5, 0.8]

        c0 = Copula('clayton')
        c0.fit(U, V)
        result = c0.get_cdf()([0, 0.1, 0.2], [0, 0.1, 0.8])
        print(result)
        expected_result = [0, 0.07517146687679954, 0.19881186077542212]
        assert result == expected_result

        c1 = Copula(cname='frank')
        c1.fit(U, V)
        result = c1.get_cdf()([0, 0.1, 0.2], [0, 0.1, 0.2])
        expected_result = np.array([-0., 0.04775112788933559, 0.13167544529480094])

        assert (result == expected_result).all()

        c2 = Copula(cname='gumbel')
        c2.fit(U, V)
        result = c2.get_cdf()([0, 0.1, 0.2], [0, 0.1, 0.8])
        expected_result = np.array([0.0, 0.042835279521916785, 0.19817042125347709])
        for i in range(3):
            assert result[i] == expected_result[i]
        assert (result == expected_result).all()

        # LOGGER.debug(Copula.select_copula(U,V))

    def test_select_copula_negative_tau(self):
        U = [0.1, 0.2, 0.3, 0.4]
        V = [0.8, 0.7, 0.6, 0.5]
        name, param = Copula.select_copula(U, V)
        assert name == 1

    def test_select_copula(self):
        U = [0.1, 0.2, 0.3, 0.4]
        V = [0.2, 0.3, 0.4, 0.5]
        name, param = Copula.select_copula(U, V)
        assert name == 1
