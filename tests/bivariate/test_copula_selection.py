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

        c0 = Copula(U, V, cname='clayton')
        result = c0.cdf([0, 0.1, 0.2], [0, 0.1, 0.8], c0.theta)
        expected_result = [0, 0.07517146687679954, 0.19881186077542212]
        assert result == expected_result

        c1 = Copula(U, V, cname='frank')
        result = c1.cdf([0, 0.1, 0.2], [0, 0.1, 0.2], c1.theta)
        expected_result = np.array([-0., 0.04775112788933559, 0.13167544529480094])

        assert (result == expected_result).all()

        c2 = Copula(U, V, cname='gumbel')
        result = c2.cdf([0, 0.1, 0.2], [0, 0.1, 0.8], c2.theta)
        expected_result = np.array([0.0, 0.042835279521916785, 0.19817042125347709])
        for i in range(3):
            assert result[i] == expected_result[i]
        assert (result == expected_result).all()

        # LOGGER.debug(Copula.select_copula(U,V))
