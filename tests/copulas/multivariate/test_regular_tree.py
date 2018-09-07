from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.multivariate.tree import Edge, Tree, TreeTypes
from copulas.univariate.kde import KDEUnivariate


class TestRegularTree(TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/iris.data.csv')
        self.tau_mat = self.data.corr(method='kendall').values
        self.u_matrix = np.empty(self.data.shape)
        count = 0
        for col in self.data:
            uni = KDEUnivariate()
            uni.fit(self.data[col])
            self.u_matrix[:, count] = [uni.cumulative_density(x) for x in self.data[col]]
            count += 1
        self.tree = Tree(TreeTypes.REGULAR)
        self.tree.fit(0, 4, self.tau_mat, self.u_matrix)

    def test_first_tree(self):
        """ Assert the construction of first tree is correct
        The first tree should be:
                   1
                0--2--3
        """
        sorted_edges = Edge.sort_edge(self.tree.edges)

        assert sorted_edges[0].L == 0
        assert sorted_edges[0].R == 2
        assert sorted_edges[1].L == 1
        assert sorted_edges[1].R == 2
        assert sorted_edges[2].L == 2
        assert sorted_edges[2].R == 3

    def test_first_tree_likelihood(self):
        """ Assert first tree likehood is correct"""
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        value, new_u = self.tree.get_likelihood(uni_matrix)

        expected = -2.2245
        assert abs(value - expected) < 10E-3

    def test_get_constraints(self):
        """ Assert get constraint gets correct neighbor nodes"""
        self.tree._get_constraints()

        assert self.tree.edges[0].neighbors == [1, 2]
        assert self.tree.edges[1].neighbors == [0, 2]

    def test_get_tau_matrix(self):
        """ Assert second tree likelihood is correct """
        self.tau = self.tree.get_tau_matrix()

        test = np.isnan(self.tau)

        self.assertFalse(test.all())

    def test_second_tree_likelihood(self):
        """ Assert second tree likelihood is correct
        """
        tau = self.tree.get_tau_matrix()
        second_tree = Tree(TreeTypes.REGULAR)
        second_tree.fit(1, 3, tau, self.tree)
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        first_value, new_u = self.tree.get_likelihood(uni_matrix)
        second_value, out_u = second_tree.get_likelihood(new_u)

        assert second_value < 0
