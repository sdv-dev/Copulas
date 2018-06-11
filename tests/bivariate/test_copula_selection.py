#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `copulas` package."""

from unittest import TestCase

from copulas.bivariate.copulas import Copula


class TestCopulas(TestCase):

    def test_snippet(self):

        U = [0.1, 0.2, 0.3, 0.4]
        V = [0.5, 0.6, 0.5, 0.8]

        c0 = Copula(U, V, cname='clayton')
        # LOGGER.debug(c0.cdf([0,0.1,0.2],[0,0.1,0.8],c0.theta))

        c1 = Copula(U, V, cname='frank')
        # LOGGER.debug(c1.cdf([0,0.1,0.2],[0,0.1,0.2],c1.theta))

        c2 = Copula(U, V, cname='gumbel')
        # LOGGER.debug(c2.cdf([0,0.1,0.2],[0,0.1,0.8],c2.theta))

        # LOGGER.debug(Copula.select_copula(U,V))
