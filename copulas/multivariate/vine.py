import logging
# from random import randint, seed, getstate, setstate
import random

import numpy as np
from scipy import optimize

from copulas import EPSILON
from copulas.bivariate.base import Bivariate, CopulaTypes
from copulas.multivariate.base import Multivariate
from copulas.multivariate.tree import Tree
from copulas.univariate.kde import KDEUnivariate

LOGGER = logging.getLogger(__name__)


class VineCopula(Multivariate):
    def __init__(self, vine_type):
        """Instantiate a vine copula class.

        Args:
            :param vine_type: type of the vine copula, could be 'cvine','dvine','rvine'
            :type vine_type: string
        """
        super().__init__()
        self.type = vine_type
        self.u_matrix = None

        self.model = KDEUnivariate

    def fit(self, X, truncated=3):
        """Fit a vine model to the data.

        Args:
            X: `np.ndarray`: data to be fitted.
            truncated: `int` max level to build the vine.
        """
        self.n_sample, self.n_var = X.shape
        self.tau_mat = X.corr(method='kendall').values
        self.u_matrix = np.empty([self.n_sample, self.n_var])

        self.truncated = truncated
        self.depth = self.n_var - 1
        self.trees = []

        self.unis, self.ppfs = [], []
        for i, col in enumerate(X):
            uni = self.model()
            uni.fit(X[col])
            self.u_matrix[:, i] = [uni.cumulative_distribution(x) for x in X[col]]
            self.unis.append(uni)
            self.ppfs.append(uni.percent_point)

        self.train_vine(self.type)

    def train_vine(self, tree_type):
        LOGGER.debug('start building tree : 0')
        tree_1 = Tree(tree_type)
        tree_1.fit(0, self.n_var, self.tau_mat, self.u_matrix)
        self.trees.append(tree_1)
        LOGGER.debug('finish building tree : 0')

        for k in range(1, min(self.n_var - 1, self.truncated)):
            # get constraints from previous tree
            self.trees[k - 1]._get_constraints()
            tau = self.trees[k - 1].get_tau_matrix()
            LOGGER.debug('start building tree: {0}'.format(k))
            tree_k = Tree(tree_type)
            tree_k.fit(k, self.n_var - k, tau, self.trees[k - 1])
            self.trees.append(tree_k)
            LOGGER.debug('finish building tree: {0}'.format(k))

    def get_likelihood(self, uni_matrix):
        """Compute likelihood of the vine."""
        num_tree = len(self.trees)
        values = np.empty([1, num_tree])

        for i in range(num_tree):
            value, new_uni_matrix = self.trees[i].get_likelihood(uni_matrix)
            uni_matrix = new_uni_matrix
            values[0, i] = value

        return np.sum(values)

    def sample(self, num_rows=1, seed=None):
        """Generating samples from vine model."""
        s1 = np.random.get_state()
        
        s2 = random.getstate()
        
        np.random.seed(seed)
        
        random.setstate(seed)
        
        unis = np.random.uniform(0, 1, self.n_var)
        
        # randomly select a node to start with
        first_ind = random.randint(0, self.n_var - 1)
        
        np.random.seed(s1)
        
        random.setstate(s2)
        
        adj = self.trees[0].get_adjacent_matrix()
        visited = []
        explore = [first_ind]

        sampled = np.zeros(self.n_var)
        itr = 0
        while explore:
            current = explore.pop(0)
            neighbors = np.where(adj[current, :] == 1)[0].tolist()
            if itr == 0:
                new_x = self.ppfs[current](unis[current])

            else:
                for i in range(itr - 1, -1, -1):
                    current_ind = -1

                    if i >= self.truncated:
                        continue

                    current_tree = self.trees[i].edges
                    # get index of edge to retrieve
                    for edge in current_tree:
                        if i == 0:
                            if (edge.L == current and edge.R == visited[0]) or\
                               (edge.R == current and edge.L == visited[0]):
                                current_ind = edge.index
                                break
                        else:
                            if edge.L == current or edge.R == current:
                                condition = set(edge.D)
                                condition.add(edge.L)
                                condition.add(edge.R)
                                visit_set = set(visited).add(current)

                                if condition.issubset(visit_set):
                                    current_ind = edge.index
                                break

                    if current_ind != -1:
                        # the node is not indepedent contional on visited node
                        copula_type = current_tree[current_ind].name
                        copula_para = current_tree[current_ind].param
                        cop = Bivariate(CopulaTypes(copula_type))
                        derivative = cop.get_h_function()
                        # start with last level
                        if i == itr - 1:
                            tmp = optimize.fminbound(
                                derivative, EPSILON, 1.0,
                                args=(unis[visited[0]], copula_para, unis[current])
                            )
                        else:
                            tmp = optimize.fminbound(
                                derivative, EPSILON, 1.0,
                                args=(unis[visited[0]], copula_para, tmp)
                            )

                        tmp = min(max(tmp, EPSILON), 0.99)

                new_x = self.ppfs[current](tmp)

            sampled[current] = new_x

            for s in neighbors:
                if s not in visited:
                    explore.insert(0, s)

            itr += 1
            visited.insert(0, current)

        return sampled
