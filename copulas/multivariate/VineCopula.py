import logging
import sys
from random import randint

import numpy as np
from scipy import optimize

from copulas.bivariate.copulas import Copula
from copulas.multivariate.MVCopula import MVCopula
from copulas.multivariate.Tree import CTree, DTree, RTree
from copulas.univariate.KDEUnivariate import KDEUnivariate

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
LOGGER.addHandler(ch)

c_map = {0: 'clayton', 1: 'frank', 2: 'gumbel'}
eps = np.finfo(np.float32).eps


class VineCopula(MVCopula):
    """ Class for a vine copula model """

    def __init__(self, type):
        super(VineCopula, self).__init__()
        self.type = type
        self.u_matrix = None

        self.cdf = None
        self.ppf = None

        self.model = None
        self.param = None

    def fit(self, data, truncated=3):
        """Fit a vine model to the data
        Returns:
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
        self.vine_model = []
        if self.type == 'cvine':
            self.train_vine(CTree)
        elif self.type == 'dvine':
            self.train_vine(DTree)
        elif self.type == 'rvine':
            self.train_vine(RTree)
        else:
            raise Exception('Unsupported vine copula type: ' + str(self.cname))

    def train_vine(self, tree):
        LOGGER.debug('start building tree : 0')
        tree_1 = tree(0, self.n_var, self.tau_mat, self.u_matrix)
        self.vine_model.append(tree_1)
        LOGGER.debug('finish building tree : 0')
        tree_1.print_tree()
        for k in range(1, min(self.n_var - 1, self.truncated)):
            # get constraints from previous tree'''
            self.vine_model[k - 1]._get_constraints()
            tau = self.vine_model[k - 1]._get_tau()
            LOGGER.debug('start building tree: {0}'.format(k))
            tree_k = tree(k, self.n_var - k, tau, self.vine_model[k - 1])
            self.vine_model.append(tree_k)
            LOGGER.debug('finish building tree: {0}'.format(k))
            tree_k.print_tree()

    def sample(self, num_rows=1):
        """generating samples from vine model"""
        unis = np.random.uniform(0, 1, self.n_var)
        # randomly select a node to start with
        first_ind = randint(0, self.n_var - 1)
        adj = self._get_adjacent_matrix()
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
                    current_tree = self.vine_model[i].edge_set
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

    def _get_adjacent_matrix(self):
        """Build adjacency matrix from the first tree
        """
        first_tree = self.vine_model[0].edge_set
        n = len(first_tree) + 1
        adj = np.zeros([n, n])
        for k in range(len(first_tree)):
            adj[first_tree[k].L, first_tree[k].R] = 1
            adj[first_tree[k].R, first_tree[k].L] = 1
        return adj
