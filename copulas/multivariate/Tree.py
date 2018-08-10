import logging

import numpy as np
import scipy

from copulas.bivariate.copulas import Copula

LOGGER = logging.getLogger(__name__)

c_map = {0: 'clayton', 1: 'frank', 2: 'gumbel'}
eps = np.finfo(np.float32).eps


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
            edge = self.edges[i]
            for j in edge.neighbors:
                if self.level == 1:
                    left_u = self.u_matrix[:, edge.L]
                    right_u = self.u_matrix[:, edge.R]
                else:
                    left_parent, right_parent = edge.parents
                    left_u, right_u = Edge.get_conditional_uni(left_parent, right_parent)
                tau[i, j], pvalue = scipy.stats.kendalltau(left_u, right_u)
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
            copula_name = c_map[edge.name]
            copula_theta = edge.theta
            if self.level == 1:
                left_u = self.u_matrix[:, edge.L]
                right_u = self.u_matrix[:, edge.R]
            else:
                left_parent, right_parent = edge.parents
                left_u, right_u = Edge.get_conditional_uni(left_parent, right_parent)
            # compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/du
            left_u = [x for x in left_u if x is not None]
            right_u = [x for x in right_u if x is not None]
            cop = Copula(copula_name)
            cop.fit(left_u, right_u)
            derivative = cop.get_h_function()
            left_given_right = derivative(left_u, right_u, copula_theta)
            right_given_left = derivative(right_u, left_u, copula_theta)
            # correction of 0 or 1
            left_given_right[left_given_right == 0] =\
                right_given_left[right_given_left == 0] = eps
            left_given_right[left_given_right == 1] =\
                right_given_left[right_given_left == 1] = 1 - eps
            edge.U = [left_given_right, right_given_left]

    def get_likelihood(self, uni_matrix):
        """Compute likelihood of the tree given an U matrix

        Args:
            :param uni_matrix: univariate matrix to evaluate likelihood on
            :type uni_matrix: a np.ndarray

        Returns:
            param value: likelihood value of the current tree
            param new_uni_matrix: next level onditional univariate matrix
            type value: float or int
            type new_uni_matrix: np.ndarray
        """
        uni_dim = uni_matrix.shape[1]
        num_edge = len(self.edges)
        values = np.zeros([1, num_edge])
        new_uni_matrix = np.empty([uni_dim, uni_dim])
        for i in range(num_edge):
            edge = self.edges[i]
            values[0, i], left_u, right_u =\
                np.log(edge.get_likelihood(uni_matrix))
            new_uni_matrix[edge.L, edge.R] = left_u
            new_uni_matrix[edge.R, edge.L] = right_u
        return np.sum(values), new_uni_matrix

    def __str__(self):
        template = 'L:{} R:{} D:{}'
        return '\n'.join([template.format(edge.L, edge.R, edge.D)
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
            name, theta = Copula.select_copula(self.u_matrix[:, 0],
                                               self.u_matrix[:, ind])
            new_edge = Edge(itr, 0, ind, self.tau_matrix[0, ind], name, theta)
            new_edge.level = self.level
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
            left_parent, right_parent = Edge.sort_edge(edges[anchor], edges[right])
            left_u, right_u = Edge.get_conditional_uni(left_parent, right_parent)
            name, theta = Copula.select_copula(left_u, right_u)
            [ed1, ed2, ing] =\
                Edge._identify_eds_ing(left_parent, right_parent)
            new_edge = Edge(itr, ed1, ed2, tau_sorted[itr, 1], name, theta)
            new_edge.D = ing
            new_edge.parents = [left_parent, right_parent]
            new_edge.level = self.level
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
            name, theta = Copula.select_copula(self.u_matrix[:, T1[k]],
                                               self.u_matrix[:, T1[k + 1]])
            left = min(T1[k], T1[k + 1])
            right = max(T1[k], T1[k + 1])
            new_edge = Edge(k, left, right, tau_T1[k], name, theta)
            new_edge.level = self.level
            self.edges.append(new_edge)

    def _build_kth_tree(self):
        edges = self.previous_tree.edges
        for k in range(self.n_nodes - 1):
            left_parent, right_parent = Edge.sort_edge(edges[k], edges[k + 1])
            left_u, right_u = Edge.get_conditional_uni(left_parent, right_parent)
            name, theta = Copula.select_copula(left_u, right_u)
            [ed1, ed2, ing] = Edge._identify_eds_ing(left_parent, right_parent)
            new_edge = Edge(k, ed1, ed2, self.tau_matrix[k, k + 1], name, theta)
            new_edge.D = ing
            new_edge.parents = [left_parent, right_parent]
            new_edge.level = self.level
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
            name, theta = Copula.select_copula(self.u_matrix[:, edge[0]],
                                               self.u_matrix[:, edge[1]])
            left = min(edge[0], edge[1])
            right = max(edge[0], edge[1])
            new_edge = Edge(itr, left, right, self.tau_matrix[edge[0], edge[1]],
                            name, theta)
            new_edge.level = self.level
            self.edges.append(new_edge)
            X.add(edge[1])
            itr += 1

    def _build_kth_tree(self):
        """build tree for level k
        """
        neg_tau = -1.0 * abs(self.tau_matrix)
        edges = self.previous_tree.edges
        visited = set([0])
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
            pairs = sorted(adj_set, key=lambda e: neg_tau[e[0]][e[1]])[0]
            left_parent, right_parent = Edge.sort_edge(edges[pairs[0]], edges[pairs[1]])
            [ed1, ed2, ing] =\
                Edge._identify_eds_ing(left_parent, right_parent)
            left_u, right_u = Edge.get_conditional_uni(left_parent, right_parent)
            name, theta = Copula.select_copula(left_u, right_u)
            new_edge = Edge(itr, ed1, ed2, self.tau_matrix[pairs[0], pairs[1]],
                            name, theta)
            new_edge.D = ing
            new_edge.parents = [left_parent, right_parent]
            new_edge.level = self.level
            self.edges.append(new_edge)
            visited.add(pairs[1])
            unvisited.remove(pairs[1])
            itr += 1


class Edge(object):
    def __init__(self, index, left, right, tau, copula_name, copula_theta):
        """Initialize an Edge object

        Args:
            :param index: index of the edge in the current tree
            :param left: left_node index (smaller)
            :param right: right_node index (larger)
            :param tau: tau value of the edge
            :param copula_name: name of the fitted copula class
            :param copula_theta: parameters of the fitted copula class

        """
        self.index = index
        self.level = None   # in which level of tree

        self.L = left
        self.R = right
        self.D = []  # dependence_set
        self.parents = None
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
        left, right = sorted(list(A ^ B))
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

    def get_likelihood(self, uni_matrix):
        """ Compute likelihood given a U matrix """
        if self.level == 1:
            left_u = uni_matrix[:, self.L]
            right_u = uni_matrix[:, self.R]
        else:
            left_u = uni_matrix[self.L, self.R]
            right_u = uni_matrix[self.R, self.L]
        print(left_u, right_u)
        cop = Copula(c_map[self.name])
        cop.set_params(theta=self.theta)
        value = np.sum(cop.get_pdf()(left_u, right_u))
        print(c_map[self.name])
        print(value)
        left_given_right = cop.get_h_function()(left_u, right_u, self.theta)
        right_given_left = cop.get_h_function()(right_u, left_u, self.theta)
        return value, left_given_right, right_given_left

    @staticmethod
    def sort_edge(edge1, edge2):
        """ sort edge object by end node indices"""
        if edge1.L < edge2.L:
            return edge1, edge2
        elif edge2.L < edge1.L:
            return edge2, edge1
        elif edge1.R < edge2.R:
            return edge1, edge2
        elif edge2.R < edge1.R:
            return edge2, edge1

    @staticmethod
    def get_conditional_uni(left_parent, right_parent):
        """ Identify pair univariate value from parents"""
        left, right, D = Edge._identify_eds_ing(left_parent, right_parent)
        if left_parent.L == left:
            left_u = left_parent.U[0]
        else:
            left_u = left_parent.U[1]
        if right_parent.L == right:
            right_u = right_parent.U[0]
        else:
            right_u = right_parent.U[1]
        return left_u, right_u