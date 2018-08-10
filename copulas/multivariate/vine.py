import logging
from random import randint

import numpy as np
from scipy import optimize

from copulas.bivariate.copulas import Copula
from copulas.multivariate.base import Multivariate
from copulas.multivariate.tree import CenterTree, DirectTree, RegularTree
from copulas.univariate.kde import KDEUnivariate

LOGGER = logging.getLogger(__name__)

c_map = {0: 'clayton', 1: 'frank', 2: 'gumbel'}
eps = np.finfo(np.float32).eps


class VineCopula(Multivariate):
    def __init__(self, type):
        super(VineCopula, self).__init__()
        """Instantiate a vine copula class

        Args:
            :param type: type of the vine copula, could be 'cvine','dvine','rvine'
            :type type: string
        """
        self.type = type
        self.u_matrix = None

        self.cdf = None
        self.ppf = None

        self.model = None
        self.param = None

    def fit(self, data, truncated=3):
        """Fit a vine model to the data

        Args:
            :param data: data to be fitted
            :param truncated: only construct the vine up to level (truncated)
            :type data: pandas DataFrame
            :type truncated: int
        """
        self.data = data
        self.n_sample = self.data.shape[0]
        self.n_var = self.data.shape[1]
        self.tau_mat = self.data.corr(method='kendall').as_matrix()
        self.u_matrix = np.empty([self.n_sample, self.n_var])
        self.unis, self.ppfs = [], []
        count = 0
        for col in data:
            uni = KDEUnivariate()
            uni.fit(data[col])
            self.u_matrix[:, count] = [uni.get_cdf(x) for x in data[col]]
            self.unis.append(uni)
            self.ppfs.append(uni.get_ppf)
            count += 1
        self.truncated = truncated
        self.depth = self.n_var - 1
        self.trees = []
        if self.type == 'cvine':
            self.train_vine(CenterTree)
        elif self.type == 'dvine':
            self.train_vine(DirectTree)
        elif self.type == 'rvine':
            self.train_vine(RegularTree)
        else:
            raise Exception('Unsupported vine copula type: ' + str(self.cname))

    def train_vine(self, tree):
        LOGGER.debug('start building tree : 0')
        tree_1 = tree(0, self.n_var, self.tau_mat, self.u_matrix)
        self.trees.append(tree_1)
        LOGGER.debug('finish building tree : 0')
        print(tree_1)
        for k in range(1, min(self.n_var - 1, self.truncated)):
            # get constraints from previous tree'''
            self.trees[k - 1]._get_constraints()
            tau = self.trees[k - 1].get_tau_matrix()
            LOGGER.debug('start building tree: {0}'.format(k))
            tree_k = tree(k, self.n_var - k, tau, self.trees[k - 1])
            self.trees.append(tree_k)
            LOGGER.debug('finish building tree: {0}'.format(k))
            print(tree_k)

    def get_likelihood(self, uni_matrix):
        """Compute likelihood of the vine"""
        num_tree = len(self.trees)
        values = np.empty([1, num_tree])
        for i in range(num_tree):
            value, new_uni_matrix = self.trees[i].get_likelihood(uni_matrix)
            uni_matrix = new_uni_matrix
            values[0, i] = value
        return np.sum(values)

    def sample(self, num_rows=1):
        """generating samples from vine model"""
        unis = np.random.uniform(0, 1, self.n_var)
        # randomly select a node to start with
        first_ind = randint(0, self.n_var - 1)
        adj = self.trees[0].get_adjacent_matrix()
        visited, explore = [], []
        explore.insert(0, first_ind)
        sampled = [0] * self.n_var
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
                        cop = Copula(c_map[copula_type])
                        derivative = cop.get_h_function()
                        # start with last level
                        if i == itr - 1:
                            tmp = optimize.fminbound(derivative, eps, 1.0,
                                                     args=(unis[visited[0]],
                                                           copula_para,
                                                           unis[current]))
                        else:
                            tmp = optimize.fminbound(derivative, eps, 1.0,
                                                     args=(unis[visited[0]],
                                                           copula_para, tmp))
                        tmp = min(max(tmp, eps), 0.99)
                new_x = self.ppfs[current](tmp)
            sampled[current] = new_x
            for s in neighbors:
                if s in visited:
                    continue
                else:
                    explore.insert(0, s)
            itr += 1
            visited.insert(0, current)
            return sampled
