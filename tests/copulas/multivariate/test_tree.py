from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from copulas.bivariate import CopulaTypes
from copulas.multivariate.tree import Edge, Tree, TreeTypes
from copulas.univariate.kde import KDEUnivariate
from tests import compare_nested_dicts, compare_nested_iterables


class TestTree(TestCase):

    def test_to_dict_unfitted_model(self):
        """to_dict returns a simpler dict when Tree is not fitted."""
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
                    'index': 0,
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
                    'index': 1,
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

    def test_get_tau_matrix_no_edges_empty(self):
        """get_tau_matrix returns an empty array if there are no edges."""
        # Setup
        tree = Tree(TreeTypes.DIRECT)
        tree.edges = []

        # Run
        result = tree.get_tau_matrix()

        # Check
        assert result.shape == (0, 0)

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
        """_identify_eds_ing returns the left, right and dependency for a new edge."""
        # Setup
        first = Edge(None, 2, 5, None, None)
        first.D = {1, 3}

        second = Edge(None, 3, 4, None, None)
        second.D = {1, 5}

        # Please, note that we passed the index, copula_name and copula_theta as None
        # To show they are no going to be used in the scope of this test.

        # left, right and dependence set
        expected_result = (2, 4, set([1, 3, 5]))

        # Run
        result = Edge._identify_eds_ing(first, second)

        # Check
        assert result == expected_result

    def test_identify_eds_empty_dependence(self):
        """_identify_eds_ing don't require edges to have dependence set."""
        # Setup
        first = Edge(None, 0, 1, None, None)
        second = Edge(None, 1, 2, None, None)

        # Please, note that we passed the index, copula_name and copula_theta as None
        # To show they are no going to be used in the scope of this test.

        # left, right and dependence set
        expected_result = (0, 2, {1})

        # Run
        result = Edge._identify_eds_ing(first, second)

        # Check
        assert result == expected_result

    def test_identify_eds_not_adjacent(self):
        """_identify_eds_ing raises a ValueError if the edges are not adjacent."""
        # Setup
        first = Edge(None, 0, 1, None, None)
        second = Edge(None, 2, 3, None, None)

        # Please, note that we passed the index, copula_name and copula_theta as None
        # To show they are no going to be used in the scope of this test.

        # Run / Check
        # As they are not adjacent, we can asure calling _identify_eds_ing will raise a ValueError.
        assert not first.is_adjacent(second)

        with self.assertRaises(ValueError):
            Edge._identify_eds_ing(first, second)

    @patch('copulas.multivariate.tree.Edge._identify_eds_ing')
    def test_get_conditional_uni(self, adjacent_mock):
        """get_conditional_uni return the corresponding univariate adjacent to the parents."""
        # Setup
        left = Edge(None, 1, 2, None, None)
        left.U = np.array([
            ['left_0_0', 'left_0_1'],
            ['left_1_0', 'left_1_1']
        ])

        right = Edge(None, 4, 2, None, None)
        right.U = np.array([
            ['right_0_0', 'right_0_1'],
            ['right_1_0', 'right_1_1']
        ])

        adjacent_mock.return_value = (0, 1, None)
        expected_result = (
            np.array(['left_1_0', 'left_1_1']),
            np.array(['right_1_0', 'right_1_1'])
        )

        # Run
        result = Edge.get_conditional_uni(left, right)

        # Check
        compare_nested_iterables(result, expected_result)

    def test_sort_edge(self):
        """sort_edge sorts Edge objects by left node index, and in case of match by right index."""
        # Setup
        edge_1 = Edge(None, 1, 0, None, None)
        edge_2 = Edge(None, 1, 1, None, None)
        edge_3 = Edge(None, 1, 2, None, None)
        edge_4 = Edge(None, 2, 0, None, None)
        edge_5 = Edge(None, 3, 0, None, None)

        edges = [edge_3, edge_1, edge_5, edge_4, edge_2]
        expected_result = [edge_1, edge_2, edge_3, edge_4, edge_5]

        # Run
        result = Edge.sort_edge(edges)

        # Check
        assert result == expected_result

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

    @patch('copulas.multivariate.tree.Bivariate', autospec=True)
    def test_get_likelihood_no_parents(self, bivariate_mock):
        """get_likelihood will use current node indices if there are no parents."""
        # Setup
        index = 0
        left = 0
        right = 1
        copula_name = 'copula_name'
        copula_theta = 'copula_theta'
        instance = Edge(index, left, right, copula_name, copula_theta)

        univariates = np.array([
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25]
        ]).T

        instance_mock = bivariate_mock.return_value
        instance_mock.probability_density.return_value = [0]
        instance_mock.partial_derivative.return_value = 'partial_derivative'

        expected_partial_derivative_call_args = [
            (
                (np.array([[
                    [0.25, 0.75],
                    [0.50, 0.50],
                ]]),), {}
            ),
            (
                (np.array([[
                    [0.50, 0.50],
                    [0.25, 0.75]
                ]]), ), {}
            )
        ]

        # Run
        result = instance.get_likelihood(univariates)

        # Check
        value, left_given_right, right_given_left = result
        assert value == 0
        assert left_given_right == 'partial_derivative'
        assert right_given_left == 'partial_derivative'

        bivariate_mock.assert_called_once_with('copula_name')

        assert instance_mock.theta == 'copula_theta'
        compare_nested_iterables(
            instance_mock.partial_derivative.call_args_list,
            expected_partial_derivative_call_args
        )

    @patch('copulas.multivariate.tree.Bivariate', autospec=True)
    def test_get_likelihood_with_parents(self, bivariate_mock):
        """If edge has parents, their dependences are used to retrieve univariates."""
        # Setup
        index = None
        left = 0
        right = 1
        copula_name = 'copula_name'
        copula_theta = 'copula_theta'
        instance = Edge(index, left, right, copula_name, copula_theta)
        instance.D = {0, 1, 2, 3}

        parent_1 = MagicMock(spec=Edge)
        parent_1.D = {1, 2, 3}

        parent_2 = MagicMock(spec=Edge)
        parent_2.D = {0, 2, 3}

        univariates = np.array([
            [0.25, 0.75],
            [0.50, 0.50],
            [0.75, 0.25]
        ]).T

        instance_mock = bivariate_mock.return_value
        instance_mock.probability_density.return_value = [0]
        instance_mock.partial_derivative.return_value = 'partial_derivative'

        expected_partial_derivative_call_args = [
            (
                (np.array([[
                    [0.25, 0.75],
                    [0.50, 0.50],
                ]]),), {}
            ),
            (
                (np.array([[
                    [0.50, 0.50],
                    [0.25, 0.75]
                ]]), ), {}
            )
        ]

        # Run
        result = instance.get_likelihood(univariates)

        # Check
        value, left_given_right, right_given_left = result
        assert value == 0
        assert left_given_right == 'partial_derivative'
        assert right_given_left == 'partial_derivative'

        bivariate_mock.assert_called_once_with('copula_name')

        assert instance_mock.theta == 'copula_theta'
        compare_nested_iterables(
            instance_mock.partial_derivative.call_args_list,
            expected_partial_derivative_call_args
        )
