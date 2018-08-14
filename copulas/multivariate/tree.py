import logging

import numpy as np
import scipy

from copulas.bivariate.base import Bivariate, CopulaTypes
from copulas.utils import EPSILON

LOGGER = logging.getLogger(__name__)


class Tree(object):
    """Helper class to instantiate a single tree in the vine model
    """
    def __init__(self, index, n_nodes, tau_matrix, previous_tree):
        """Initialize tree object

        Args:
            :param index: index of the tree
            :param n_nodes: number of nodes in the tree
            :tau_matrix: kendall's tau matrix of the data
            :previous_tree: tree object of previous level
            :type index: int
            :type n_nodes: int
            :type tau_matrix: np.ndarray of size n_nodes*n_nodes
        """
        self.level = index + 1
        self.previous_tree = previous_tree
        self.n_nodes = n_nodes
        self.edges = []
        self.tau_matrix = tau_matrix
        if self.level == 1:
            self.u_matrix = previous_tree
            self._build_first_tree()
        else:
            self._build_kth_tree()
        self.prepare_next_tree()

    def _check_contraint(self, edge1, edge2):
        """Check if two edges satisfy vine constraint

        Args:
            :param edge1: edge object representing edge1
            :param edge2: edge object representing edge2
            :type edge1: Edge object
            :type edge2: Edge object

        Returns:
            Boolean True if the two edges satisfy vine constraints
        """
        full_node = set([edge1.L, edge1.R, edge2.L, edge2.R])
        full_node.update(edge1.D)
        full_node.update(edge2.D)
        return len(full_node) == (self.level + 1)

    def _get_constraints(self):
        """Get neighboring edges for each edge in the edges"""
        num_edges = len(self.edges)
        for k in range(num_edges):
            for i in range(num_edges):
                # add to constriants if i shared an edge with k
                if k != i:
                    if self.edges[k].is_adjacent(self.edges[i]):
                        self.edges[k].neighbors.append(i)

    def get_tau_matrix(self):
        """Get tau matrix for adjacent pairs

        Returns:
            :param tau: tau matrix for the current tree
            :type tau: np.ndarray
        """
        num_edges = len(self.edges)
        tau = np.empty([num_edges, num_edges])
        for i in range(num_edges):
            for j in self.edges[i].neighbors:
                left_parent, right_parent = self.edges[i].parent
                if self.level == 1:
                    U1 = self.u_matrix[:, left_parent]
                    U2 = self.u_matrix[:, right_parent]
                else:
                    U1 = self.previous_tree.edges[left_parent].U[0]
                    U2 = self.previous_tree.edges[right_parent].U[0]
                tau[i, j], pvalue = scipy.stats.kendalltau(U1, U2)
        return tau

    def get_adjacent_matrix(self):
        """ Get adjacency matrix

        Returns:
            :param adj: adjacency matrix
            :type adj: np.ndarray
        """
        edges = self.edges
        num_edges = len(edges) + 1
        adj = np.zeros([num_edges, num_edges])
        for k in range(num_edges - 1):
            adj[edges[k].L, edges[k].R] = 1
            adj[edges[k].R, edges[k].L] = 1
        return adj

    def prepare_next_tree(self):
        """Prepare conditional U matrix for next tree
        """
        num_edges = len(self.edges)
        for k in range(num_edges):
            edge = self.edges[k]
            copula_name = CopulaTypes(edge.name)
            copula_theta = edge.theta
            if self.level == 1:
                U1 = self.u_matrix[:, edge.L]
                U2 = self.u_matrix[:, edge.R]
            else:
                previous_tree = self.previous_tree.edges
                left_parent, right_parent = edge.parent
                U1 = previous_tree[left_parent].U[0]
                U2 = previous_tree[right_parent].U[0]
            # compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/du
            U1 = [x for x in U1 if x is not None]
            U2 = [x for x in U2 if x is not None]
            c1 = Bivariate(copula_name)
            c1.fit(U1, U2)
            U1_given_U2 = c1.partial_derivative_cumulative_density(U2, U1, copula_theta)
            U2_given_U1 = c1.partial_derivative_cumulative_density(U1, U2, copula_theta)
            # correction of 0 or 1
            U1_given_U2[U1_given_U2 == 0] = U2_given_U1[U2_given_U1 == 0] = EPSILON
            U1_given_U2[U1_given_U2 == 1] = U2_given_U1[U2_given_U1 == 1] = 1 - EPSILON
            edge.U = [U1_given_U2, U2_given_U1]

    def get_likelihood(self, uni_matrix):
        """Compute likelihood of the tree given an U matrix

        Args:
            :param uni_matrix: univariate matrix to evaluate likelihood on
            :type uni_matrix: a np.ndarray

        Returns:
            param value: likelihood value of the current tree
            type value: float or int
        """
        edges = self.edges
        values = np.zeros([1, len(edges)])
        for i in range(len(edges)):
            edge = edges[i]
            cname = CopulaTypes(edge.name)
            v1, v2 = edge.L, edge.R
            copula_theta = edge.theta
            if self.level == 1:
                U1 = np.array([uni_matrix[v1]])
                U2 = np.array([uni_matrix[v2]])
            else:
                previous_tree = self.previous_tree.edges
                left_parent, right_parent = edge.parent
                U1 = previous_tree[left_parent].U[0]
                U2 = previous_tree[right_parent].U[0]
            cop = Bivariate(cname)
            cop.fit(U1, U2)
            values[0, i] = cop.pdf()(U1, U2)
            U1_given_U2 = cop.get_h_function()(U2, U1, copula_theta)
            # U2givenU1 = derivative(U1, U2, copula_theta)
            edge.U = U1_given_U2
            value = np.sum(np.log(values))
        return value

    def __str__(self):
        template = 'L:{} R:{} D:{} parent:{}'
        return '\n'.join([template.format(edge.L, edge.R, edge.D, edge.parent)
                          for edge in self.edges])


class CenterTree(Tree):
    """Helper Class for instantiate a Center Vine"""
    def _build_first_tree(self):
        """build first level tree"""
        # first column is the variable of interest
        np.fill_diagonal(self.tau_matrix, np.NaN)
        tau_y = self.tau_matrix[:, 0]
        temp = np.empty([self.n_nodes, 3])
        temp[:, 0] = np.arange(self.n_nodes)
        temp[:, 1] = tau_y
        temp[:, 2] = abs(tau_y)
        temp[np.isnan(temp)] = -10
        tau_sorted = temp[temp[:, 2].argsort()[::-1]]
        for itr in range(1, self.n_nodes):
            ind = tau_sorted[itr, 0]
            name, theta = Bivariate.select_copula(self.u_matrix[:, 0], self.u_matrix[:, ind])
            new_edge = Edge(itr, 0, ind, self.tau_matrix[0, ind], name, theta)
            new_edge.parent.append(0)
            new_edge.parent.append(ind)
            self.edges.append(new_edge)

    def _build_kth_tree(self):
        """build k-th level tree"""
        anchor, tau_sorted = self.get_anchor()
        self.tau_matrix[anchor, :] = np.NaN
        # sort the rest of variables based on dependence with anchor variable
        aux = np.empty([self.n_nodes, 3])
        aux[:, 0] = np.arange(self.n_nodes, dtype=int)
        aux[:, 1] = self.tau_matrix[:, anchor]
        aux[:, 2] = abs(self.tau_matrix[:, anchor])
        aux[anchor, 2] = -10
        aux_sorted = aux[aux[:, 2].argsort()[::-1]]
        edges = self.previous_tree.edges
        for itr in range(self.n_nodes - 1):
            right = aux_sorted[itr, 0]
            U1, U2 = edges[anchor].U[0], edges[right].U[0]
            name, theta = Bivariate.select_copula(U1, U2)
            [ed1, ed2, ing] = Edge._identify_eds_ing(edges[anchor], edges[right])
            new_edge = Edge(itr, ed1, ed2, tau_sorted[itr, 1], name, theta)
            new_edge.D = ing
            new_edge.parents.append(edges[anchor])
            new_edge.parents.append(edges[right])
            new_edge.parent.append(anchor)
            new_edge.parent.append(right)
            self.edges.append(new_edge)

    def get_anchor(self):
        """find anchor variable with highest sum of dependence with the rest"""
        temp = np.empty([self.n_nodes, 2])
        temp[:, 0] = np.arange(self.n_nodes, dtype=int)
        temp[:, 1] = np.sum(abs(self.tau_matrix), 1)
        tau_sorted = temp[temp[:, 1].argsort()[::-1]]
        anchor = int(temp[0, 0])
        return anchor, tau_sorted


class DirectTree(Tree):
    """Helper Class for instantiate a Direct Vine"""
    def _build_first_tree(self):
        # find the pair of maximum tau
        tau_matrix = self.tau_matrix
        np.fill_diagonal(tau_matrix, np.NaN)
        tau_y = tau_matrix[:, 0]
        temp = np.empty([self.n_nodes, 3])
        temp[:, 0] = np.arange(self.n_nodes)
        temp[:, 1] = tau_y
        temp[:, 2] = abs(tau_y)
        temp[np.isnan(temp)] = -10
        tau_sorted = temp[temp[:, 2].argsort()[::-1]]
        left_ind = tau_sorted[0, 0]
        right_ind = tau_sorted[1, 0]
        T1 = np.array([left_ind, 0, right_ind]).astype(int)
        tau_T1 = tau_sorted[:2, 1]
        # replace tau matrix of the selected variables as a negative number
        tau_matrix[:, [T1]] = -10
        for k in range(2, self.n_nodes - 1):
            valL, left = np.max(tau_matrix[T1[0], :]),\
                np.argmax(tau_matrix[T1[0], :])
            valR, right = np.max(tau_matrix[T1[-1], :]),\
                np.argmax(tau_matrix[T1[-1], :])
            if valL > valR:
                # add nodes to the left
                T1 = np.append(int(left), T1)
                tau_T1 = np.append(valL, tau_T1)
                tau_matrix[:, left] = -10
            else:
                # add node to the right
                T1 = np.append(T1, int(right))
                tau_T1 = np.append(tau_T1, valR)
                tau_matrix[:, right] = -10
        for k in range(self.n_nodes - 1):
            name, theta = Bivariate.select_copula(
                self.u_matrix[:, T1[k]], self.u_matrix[:, T1[k + 1]])

            new_edge = Edge(k, T1[k], T1[k + 1], tau_T1[k], name, theta)
            new_edge.parent.append(T1[k])
            new_edge.parent.append(T1[k + 1])
            self.edges.append(new_edge)

    def _build_kth_tree(self):
        edges = self.previous_tree.edges
        for k in range(self.n_nodes - 1):
            left_parent = edges[k]
            right_parent = edges[k + 1]
            name, theta = Bivariate.select_copula(left_parent.U[0], right_parent.U[0])
            [ed1, ed2, ing] = Edge._identify_eds_ing(left_parent, right_parent)
            new_edge = Edge(k, ed1, ed2, self.tau_matrix[k, k + 1], name, theta)
            new_edge.D = ing
            new_edge.parent.append(k)
            new_edge.parent.append(k + 1)
            self.edges.append(new_edge)


class RegularTree(Tree):
    """ Helper class for instantiate Regular Vine"""
    def _build_first_tree(self):
        """build the first tree with n-1 variable"""
        # Prim's algorithm
        neg_tau = -1.0 * abs(self.tau_matrix)
        X = set()
        X.add(0)
        itr = 0
        while len(X) != self.n_nodes:
            adj_set = set()
            for x in X:
                for k in range(self.n_nodes):
                    if k not in X and k != x:
                        adj_set.add((x, k))
            # find edge with maximum
            edge = sorted(adj_set, key=lambda e: neg_tau[e[0]][e[1]])[0]
            name, theta = Bivariate.select_copula(
                self.u_matrix[:, edge[0]], self.u_matrix[:, edge[1]])

            tau_matrix = self.tau_matrix[edge[0], edge[1]]
            new_edge = Edge(itr, edge[0], edge[1], tau_matrix, name, theta)
            new_edge.parent.append(edge[0])
            new_edge.parent.append(edge[1])
            self.edges.append(new_edge)
            X.add(edge[1])
            itr += 1

    def _build_kth_tree(self):
        """build tree for level k
        """
        neg_tau = -1.0 * abs(self.tau_matrix)
        edges = self.previous_tree.edges
        visited = set()
        visited.add(0)
        unvisited = set(range(self.n_nodes))
        itr = 0
        while len(visited) != self.n_nodes:
            adj_set = set()
            for x in visited:
                for k in range(self.n_nodes):
                    if k not in visited and k != x:
                        # check if (x,k) is a valid edge in the vine
                        if self._check_contraint(edges[x], edges[k]):
                            adj_set.add((x, k))
            # find edge with maximum tau
            if len(list(adj_set)) == 0:
                visited.add(list(unvisited)[0])
                continue
            edge = sorted(adj_set, key=lambda e: neg_tau[e[0]][e[1]])[0]
            [ed1, ed2, ing] =\
                Edge._identify_eds_ing(edges[edge[0]], edges[edge[1]])
            left_parent, right_parent = edge
            U1, U2 = edges[left_parent].U[0], edges[right_parent].U[0]
            name, theta = Bivariate.select_copula(U1, U2)
            new_edge = Edge(itr, ed1, ed2, self.tau_matrix[edge[0], edge[1]],
                            name, theta)
            new_edge.D = ing
            new_edge.parent.append(edge[0])
            new_edge.parent.append(edge[1])
            new_edge.parents.append(edges[edge[0]])
            new_edge.parents.append(edges[edge[1]])
            # new_edge.likelihood = np.log(cop.pdf(U1,U2,theta))
            self.edges.append(new_edge)
            visited.add(edge[1])
            unvisited.remove(edge[1])
            itr += 1


class Edge(object):
    def __init__(self, index, left, right, tau, copula_name, copula_theta):
        """Initialize an Edge object

        Args:
            :param index: index of the edge in the current tree
            :param left: left_node index
            :param right: right_node index
            :param tau: tau value of the edge
            :param copula_name: name of the fitted copula class
            :param copula_theta: parameters of the fitted copula class

        """
        self.index = index
        self.level = None   # in which level of tree

        self.L = left
        self.R = right
        self.D = []  # dependence_set
        self.parent = []  # indices of parent edges in the previous tree
        self.parents = []
        self.neighbors = []

        self.tau = tau
        self.name = copula_name
        self.theta = copula_theta
        self.U = None
        self.likelihood = None

    @staticmethod
    def _identify_eds_ing(edge1, edge2):
        """Find nodes connecting adjacent edges

        Args:
            :param edge1: edge object representing edge1
            :param edge2: edge object representing edge2
            :type edge1: Edge object
            :type edge2: Edge object
            :return left,right: end node indices of the new edge
            :return D: dependence_set of the new edge
            :type left, right: int
            :type D: list
        """
        A = set([edge1.L, edge1.R])
        A.update(edge1.D)
        B = set([edge2.L, edge2.R])
        B.update(edge2.D)
        D = list(A & B)
        left, right = list(A ^ B)
        return left, right, D

    def is_adjacent(self, another_edge):
        """Check if two edges are adjacent

        Args:
            :param another_edge: edge object of another edge
            :type another_edge: edge object

        This function will return true if the two edges are adjacent
        """
        return (self.L == another_edge.L or self.L == another_edge.R or
                self.R == another_edge.L or self.R == another_edge.R)

    def get_likelihood(self, U):
        """ Compute likelihood given a U matrix """
        if self.level == 1:
            name, theta = Bivariate.select_copula(U[:, self.L], U[:, self.R])
            cop = Bivariate(name)
            cop.fit(U[:, self.L], U[:, self.R])
            pdf = cop.get_pdf()
            self.likelihood = pdf(U[:, self.L], U[:, self.R])
        else:
            left_parent, right_parent = self.parents
            U1, U2 = left_parent.U[0], right_parent.U[0]
            name, theta = Bivariate.select_copula(U1, U2)
            cop = Bivariate(name)
            cop.fit(U1, U2)
            pdf = cop.get_pdf()
            self.likelihood = pdf(U1, U2)
