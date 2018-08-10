import logging
from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.multivariate.Tree import DirectTree, Edge, RegularTree
from copulas.univariate.KDEUnivariate import KDEUnivariate

LOGGER = logging.getLogger(__name__)


class TestCenterTree(TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/example.csv')
        self.tau_mat = self.data.corr(method='kendall').as_matrix()
        self.u_matrix = np.empty([self.data.shape[0], self.data.shape[1]])
        count = 0
        for col in self.data:
            uni = KDEUnivariate()
            uni.fit(self.data[col])
            self.u_matrix[:, count] = [uni.get_cdf(x) for x in self.data[col]]
            count += 1
        self.tree = DirectTree(0, 3, self.tau_mat, self.u_matrix)

    def test_identify_eds(self):
        e1 = Edge(0, 1, 5, 0.3, 'clayton', 1.5)
        e1.D = [3, 4]
        e2 = Edge(0, 2, 5, 0.3, 'clayton', 1.5)
        e2.D = [3, 4]

        left, right, D = Edge._identify_eds_ing(e1, e2)

        expected = [3, 4, 5]
        self.assertEquals(left, 1)
        self.assertEquals(right, 2)
        self.assertEquals(D, expected)

    def test_first_tree(self):
        self.tree._build_first_tree()
        self.assertEquals(self.tree.edges[0].L, 0)


class TestRegularTree(TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/example.csv')
        self.tau_mat = self.data.corr(method='kendall').as_matrix()
        self.u_matrix = np.empty([self.data.shape[0], self.data.shape[1]])
        count = 0
        for col in self.data:
            uni = KDEUnivariate()
            uni.fit(self.data[col])
            self.u_matrix[:, count] = [uni.get_cdf(x) for x in self.data[col]]
            count += 1
        self.trees = []
        self.trees.append(RegularTree(0, 3, self.tau_mat, self.u_matrix))

    def test_likelihood(self):
        uni_matrix = np.array([[1, 2, 3]])
        value, new_u = self.trees[0].get_likelihood(uni_matrix)
        self.assertAlmostEquals(value, -16.6351, places=3)

    def test_get_constraints(self):
        first_tree = self.trees[0]
        first_tree._get_constraints()
        self.assertEquals(first_tree.edges[0].neighbors, [1])
        self.assertEquals(first_tree.edges[1].neighbors, [0])

    def test_get_tau_matrix(self):
        self.tau = self.trees[0].get_tau_matrix()
        test = np.isnan(self.tau)
        self.assertFalse(test.all())

    # def test_second_tree(self):
    #     tau = self.trees[0].get_tau_matrix()
    #     second_tree = RegularTree(1, 2, tau, self.trees[0])
    #     print(second_tree)
    #     uni_matrix = np.array([[1, 2, 3]])
    #     first_value, new_u = self.trees[0].get_likelihood(uni_matrix)
    #     second_value, new_u = second_tree.get_likelihood(new_u)
    #     self.assertAlmostEquals(second_value, -17.328, places=3)
