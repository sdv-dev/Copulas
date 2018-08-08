from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.multivariate.Tree import DTree, Edge
from copulas.univariate.KDEUnivariate import KDEUnivariate


class TestTree(TestCase):
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
        self.dtree = DTree(0, 3, self.tau_mat, self.u_matrix)

    def test_identify_eds(self):
        e1 = Edge(0, 1, 5, 0.3, 'clayton', 1.5)
        e1.D = [3, 4]
        e2 = Edge(0, 2, 5, 0.3, 'clayton', 1.5)
        e2.D = [3, 4]

        left, right, D = self.dtree._identify_eds_ing(e1, e2)

        expected = [3, 4, 5]
        self.assertEquals(left, 1)
        self.assertEquals(right, 2)
        self.assertEquals(D, expected)

    def test_first_tree(self):
        # self.ctree._build_first_tree()
        # for edge in self.ctree.edge_set:
        #     self.assertEquals(0, edge.L)

        self.dtree._build_first_tree()
        self.assertEquals(self.dtree.edge_set[0].L, 2)
