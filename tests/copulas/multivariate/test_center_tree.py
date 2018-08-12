import logging
from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.multivariate.tree import CenterTree
from copulas.univariate.kde import KDEUnivariate

LOGGER = logging.getLogger(__name__)


class TestCenterTree(TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/iris.data.csv')
        self.tau_mat = self.data.corr(method='kendall').values
        self.u_matrix = np.empty([self.data.shape[0], self.data.shape[1]])
        count = 0
        for col in self.data:
            uni = KDEUnivariate()
            uni.fit(self.data[col])
            self.u_matrix[:, count] = [uni.get_cdf(x) for x in self.data[col]]
            count += 1
        self.trees = []
        self.trees.append(CenterTree(0, 4, self.tau_mat, self.u_matrix))

    def test_first_tree(self):
        self.assertEquals(self.trees[0].edges[0].L, 0)

    def test_first_tree_likelihood(self):
        print(self.trees[0])
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])
        value, new_u = self.trees[0].get_likelihood(uni_matrix)
        self.assertAlmostEquals(value, -5.5411, places=3)

    def test_get_constraints(self):
        first_tree = self.trees[0]
        first_tree._get_constraints()
        self.assertEquals(first_tree.edges[0].neighbors, [1, 2])
        self.assertEquals(first_tree.edges[1].neighbors, [0, 2])

    def test_get_tau_matrix(self):
        self.tau = self.trees[0].get_tau_matrix()
        test = np.isnan(self.tau)
        self.assertFalse(test.all())

    def test_second_tree_likelihood(self):
        tau = self.trees[0].get_tau_matrix()
        second_tree = CenterTree(1, 3, tau, self.trees[0])
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])
        first_value, new_u = self.trees[0].get_likelihood(uni_matrix)
        second_value, out_u = second_tree.get_likelihood(new_u)
        self.assertAlmostEquals(second_value, 2.4927, places=3)
