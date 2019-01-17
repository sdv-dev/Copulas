from unittest import TestCase

import numpy as np
import pandas as pd

from copulas.bivariate import CopulaTypes
from copulas.multivariate.tree import Edge, Tree, TreeTypes
from copulas.univariate.kde import KDEUnivariate
from tests import compare_nested_dicts


class TestTree(TestCase):

    def test_to_dict_unfitted_model(self):
        # Setup
        instance = Tree(TreeTypes.REGULAR)
        expected_result = {
            'type': 'copulas.multivariate.tree.RegularTree',
            'tree_type': TreeTypes.REGULAR,
            'fitted': False
        }

        # Run
        result = instance.to_dict()

        # Check
        assert result == expected_result

    def test_to_dict_fit_model(self):
        # Setup
        instance = Tree(TreeTypes.REGULAR)
        X = pd.DataFrame(data=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        index = 0
        n_nodes = X.shape[1]
        tau_matrix = X.corr(method='kendall').values

        univariates_matrix = np.empty(X.shape)
        for i, column in enumerate(X):
            distribution = KDEUnivariate()
            distribution.fit(X[column])
            univariates_matrix[:, i] = [distribution.cumulative_distribution(x) for x in X[column]]

        instance.fit(index, n_nodes, tau_matrix, univariates_matrix)
        expected_result = {
            'type': 'copulas.multivariate.tree.RegularTree',
            'fitted': True,
            'level': 1,
            'n_nodes': 3,
            'previous_tree': [
                [0.8230112726144534, 0.3384880496294825, 0.3384880496294825],
                [0.3384880496294825, 0.8230112726144534, 0.3384880496294825],
                [0.3384880496294825, 0.3384880496294825, 0.8230112726144534]
            ],
            'tau_matrix': [
                [1.0, -0.49999999999999994, -0.49999999999999994],
                [-0.49999999999999994, 1.0, -0.49999999999999994],
                [-0.49999999999999994, -0.49999999999999994, 1.0]
            ],
            'tree_type': TreeTypes.REGULAR,
            'edges': [
                {
                    'D': set(),
                    'L': 0,
                    'R': 1,
                    'U': [
                        [6.533235975920359, 6.425034969827687, 5.857062027493768],
                        [6.425034969827687, 6.533235975920359, 5.857062027493768]
                    ],
                    'likelihood': None,
                    'name': CopulaTypes.FRANK,
                    'neighbors': [],
                    'parents': None,
                    'tau': -0.49999999999999994,
                    'theta': -5.736282443655552
                },
                {
                    'D': set(),
                    'L': 1,
                    'R': 2,
                    'U': [
                        [5.857062027493768, 6.533235975920359, 6.425034969827687],
                        [5.857062027493768, 6.425034969827687, 6.533235975920359]
                    ],
                    'likelihood': None,
                    'name': CopulaTypes.FRANK,
                    'neighbors': [],
                    'parents': None,
                    'tau': -0.49999999999999994,
                    'theta': -5.736282443655552
                }
            ],
        }

        # Run
        result = instance.to_dict()

        # Check
        compare_nested_dicts(result, expected_result)

    def test_from_dict_unfitted_model(self):
        # Setup
        params = {
            'tree_type': TreeTypes.REGULAR,
            'fitted': False
        }

        # Run
        result = Tree.from_dict(params)

        # Check
        assert result.tree_type == TreeTypes.REGULAR
        assert result.fitted is False

    def test_serialization_unfitted_model(self):
        # Setup
        instance = Tree(TreeTypes.REGULAR)

        # Run
        result = Tree.from_dict(instance.to_dict())

        # Check
        assert instance.to_dict() == result.to_dict()

    def test_serialization_fit_model(self):
        # Setup
        instance = Tree(TreeTypes.REGULAR)
        X = pd.DataFrame(data=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        index = 0
        n_nodes = X.shape[1]
        tau_matrix = X.corr(method='kendall').values

        univariates_matrix = np.empty(X.shape)
        for i, column in enumerate(X):
            distribution = KDEUnivariate()
            distribution.fit(X[column])
            univariates_matrix[:, i] = [distribution.cumulative_distribution(x) for x in X[column]]

        instance.fit(index, n_nodes, tau_matrix, univariates_matrix)

        # Run
        result = Tree.from_dict(instance.to_dict())

        # Check
        assert result.to_dict() == instance.to_dict()


class TestCenterTree(TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/iris.data.csv')
        self.tau_mat = self.data.corr(method='kendall').values
        self.u_matrix = np.empty(self.data.shape)
        count = 0
        for col in self.data:
            uni = KDEUnivariate()
            uni.fit(self.data[col])
            self.u_matrix[:, count] = [uni.cumulative_distribution(x) for x in self.data[col]]
            count += 1
        self.tree = Tree(TreeTypes.CENTER)
        self.tree.fit(0, 4, self.tau_mat, self.u_matrix)

    def test_first_tree(self):
        """ Assert 0 is the center node"""
        assert self.tree.edges[0].L == 0

    def test_first_tree_likelihood(self):
        """ Assert first tree likehood is correct"""
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        value, new_u = self.tree.get_likelihood(uni_matrix)

        expected = -0.19988720707143634
        assert abs(value - expected) < 10E-3

    def test_get_constraints(self):
        """ Assert get constraint gets correct neighbor nodes"""
        self.tree._get_constraints()

        assert self.tree.edges[0].neighbors == [1, 2]
        assert self.tree.edges[1].neighbors == [0, 2]

    def test_get_tau_matrix(self):
        """ Assert none of get tau matrix is NaN """
        self.tau = self.tree.get_tau_matrix()

        test = np.isnan(self.tau)

        self.assertFalse(test.all())

    def test_second_tree_likelihood(self):
        """ Assert second tree likelihood is correct """
        tau = self.tree.get_tau_matrix()
        second_tree = Tree(TreeTypes.CENTER)
        second_tree.fit(1, 3, tau, self.tree)
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        first_value, new_u = self.tree.get_likelihood(uni_matrix)
        second_value, out_u = second_tree.get_likelihood(new_u)

        expected = 0.540089320412914
        assert abs(second_value - expected) < 10E-3


class TestRegularTree(TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/iris.data.csv')
        self.tau_mat = self.data.corr(method='kendall').values
        self.u_matrix = np.empty(self.data.shape)
        count = 0
        for col in self.data:
            uni = KDEUnivariate()
            uni.fit(self.data[col])
            self.u_matrix[:, count] = [uni.cumulative_distribution(x) for x in self.data[col]]
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

        expected = 0.9545348664739628
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
        """Assert second tree likelihood is correct."""
        tau = self.tree.get_tau_matrix()
        second_tree = Tree(TreeTypes.REGULAR)
        second_tree.fit(1, 3, tau, self.tree)
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        first_value, new_u = self.tree.get_likelihood(uni_matrix)
        second_value, out_u = second_tree.get_likelihood(new_u)

        # assert second_value < 0


class TestDirectTree(TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/iris.data.csv')
        self.tau_mat = self.data.corr(method='kendall').values
        self.u_matrix = np.empty(self.data.shape)
        count = 0
        for col in self.data:
            uni = KDEUnivariate()
            uni.fit(self.data[col])
            self.u_matrix[:, count] = [uni.cumulative_distribution(x) for x in self.data[col]]
            count += 1
        self.tree = Tree(TreeTypes.DIRECT)
        self.tree.fit(0, 4, self.tau_mat, self.u_matrix)

    def test_first_tree(self):
        """ Assert 0 is the center node"""
        assert self.tree.edges[0].L == 0

    def test_first_tree_likelihood(self):
        """ Assert first tree likehood is correct"""
        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        value, new_u = self.tree.get_likelihood(uni_matrix)

        expected = -0.1207611551427385
        assert abs(value - expected) < 10E-3

    def test_get_constraints(self):
        """ Assert get constraint gets correct neighbor nodes"""
        self.tree._get_constraints()

        assert self.tree.edges[0].neighbors == [1]
        assert self.tree.edges[1].neighbors == [0, 2]

    def test_get_tau_matrix(self):
        """ Assert none of get tau matrix is NaN """
        self.tau = self.tree.get_tau_matrix()

        test = np.isnan(self.tau)

        self.assertFalse(test.all())

    def test_second_tree_likelihood(self):
        """ Assert second tree likelihood is correct """
        tau = self.tree.get_tau_matrix()

        second_tree = Tree(TreeTypes.DIRECT)
        second_tree.fit(1, 3, tau, self.tree)

        uni_matrix = np.array([[0.1, 0.2, 0.3, 0.4]])

        first_value, new_u = self.tree.get_likelihood(uni_matrix)
        second_value, out_u = second_tree.get_likelihood(new_u)

        expected = 0.7184205492690413
        assert abs(second_value - expected) < 10E-3


class TestEdge(TestCase):
    def setUp(self):
        self.e1 = Edge(0, 2, 5, 'clayton', 1.5)
        self.e1.D = [1, 3]
        self.e2 = Edge(1, 3, 4, 'clayton', 1.5)
        self.e2.D = [1, 5]

    def test_identify_eds(self):
        left, right, depend_set = Edge._identify_eds_ing(self.e1, self.e2)
        assert left == 2
        assert right == 4
        expected_result = set([1, 3, 5])
        assert depend_set == expected_result

    def test_sort_edge(self):
        sorted_edges = Edge.sort_edge([self.e2, self.e1])
        assert sorted_edges[0].L == 2

    def test_to_dict(self):
        """To_dict returns a dictionary with the parameters to recreate an edge."""
        # Setup
        edge = Edge(1, 2, 5, 'clayton', 1.5)
        edge.D = [1, 3]
        expected_result = {
            'index': 1,
            'L': 2,
            'R': 5,
            'name': 'clayton',
            'theta': 1.5,
            'D': [1, 3],
            'U': None,
            'likelihood': None,
            'neighbors': [],
            'parents': None,
            'tau': None
        }

        # Run
        result = edge.to_dict()

        # Check
        assert result == expected_result

    def test_from_dict(self):
        """From_dict sets the dictionary values as instance attributes."""
        # Setup
        parameters = {
            'index': 0,
            'L': 2,
            'R': 5,
            'name': 'clayton',
            'theta': 1.5,
            'D': [1, 3],
            'U': None,
            'likelihood': None,
            'neighbors': [1],
            'parents': None,
            'tau': None
        }

        # Run
        edge = Edge.from_dict(parameters)

        # Check
        assert edge.index == 0
        assert edge.L == 2
        assert edge.R == 5
        assert edge.name == 'clayton'
        assert edge.theta == 1.5
        assert edge.D == [1, 3]
        assert not edge.U
        assert not edge.parents
        assert edge.neighbors == [1]

    def test_valid_serialization(self):
        # Setup
        instance = Edge(0, 2, 5, 'clayton', 1.5)

        # Run
        result = Edge.from_dict(instance.to_dict())

        # Check
        assert instance.to_dict() == result.to_dict()
