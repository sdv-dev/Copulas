from __future__ import absolute_import

import logging
import warnings
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize as optimize

from copulas import utils
from copulas.bivariate import copulas

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


class CopulaModel(object):
    """This class instantiates a Copula Model from a dataset.
    Attributes:
        data_path: A string indicating the directory of the training dataset
        meta_path: A meta file specifies information for each column
        model_data: A dataframe storing training data
        u_type: methods for estimating density, can be 'kde' or 'gaussian'
        c_type: A string indicating the type of copula models, can be
        'gaussian','frank','clayton','gumbel','cvine','dvine','t','PCA'
        y_ind : index of the variable that will be used as y variable
        n_var: number of variables

        param: A list of parameters fitted to the data for each variable
        cdfs: A list of cdf function fitted to the data for each variable
        ppfs: A list of inverse cdf function fitting to the data for each variable
        u_matrix: A matrix represents the univariate distribution of size m*n,
            where m is the number of data points, and n is number of variables
    """

    def __init__(self, data_path, utype, ctype, y_ind=None, meta_path=None):
        self.data_path = data_path
        self.meta_path = meta_path
        self.model_data = pd.read_csv(
            data_path, sep=',', index_col=False,
            na_values=['NaN', 'nan', 'NULL', 'null'], low_memory=False)
        self.u_type = utype
        self.c_type = ctype
        self.n_sample = self.model_data.shape[0]
        self.n_var = self.model_data.shape[1]
        self.y_ind = y_ind if y_ind else self.n_var - 1
        self.families = ['frank', 'clayton', 'gumbel', 'gaussian', 'cvine', 'dvine']
        LOGGER.debug('number of variables: %d', self.n_var)

        # transform copulas into its univariate
        self.cdfs, self.u_matrix, self.ppfs = self._preprocessing(self.model_data)
        self.V = self.u_matrix[:, self.y_ind]
        self.U = np.delete(self.u_matrix, self.y_ind, axis=1)
        # information about the copula model
        self.tau_mat = self.model_data.corr(method='kendall').as_matrix()
        LOGGER.debug(self.tau_mat)
        self.model = None
        self.param = None
        self._fit_model()

    def _preprocessing(self, data):
        """Preprocessing steps for the dataframe before building copulas.

        Retrieve meta files, add noise for integer columns and
        compute the cdf,ppf for each column and transform data in cdfs.

        Returns:
        cdfs: list of cdf function generator for each column
        ppfs: list of ppf function generator for each column
        unis: np matrix of data after applying cdf function to each column
        """
        cdfs = []
        ppfs = []
        unis = np.empty([data.shape[0], data.shape[1]])
        count = 0
        for col in data:
            # noise = np.random.normal(0,0.01,self.n_sample)
            # data[col]=data[col].astype('float32')
            # perturbed = noise+data[col].values
            dist = utils.Distribution(column=data[col], summary={'name': 'kde', 'values': None})
            # dist.name=self.u_type
            cdf = dist.cdf
            cdfs.append(cdf)
            unis[:, count] = [cdf(x) for x in list(data[col].values)]
            count += 1
        return cdfs, unis, ppfs

    def _fit_model(self):
        """Fit copula model to the data
        Returns:
        self.param: param of copula family, tree_data if model is a vine
        """
        if 'vine' not in self.c_type:
            if self.n_var != 2:
                msg = 'bivariate copula only support 2 variables, {} provided'.format(
                    self.u_matrix.shape[1])
                raise Exception(msg)
            else:
                cop = copulas.Copula(self.V, self.U, name=self.c_type)
                self.param = cop.theta
        elif self.c_type == 'cvine':
            vine = Vine(self.c_type, self.u_matrix, self.tau_mat, self.ppfs, self.y_ind)
            self.model = vine
            self.param = vine.vine_model
        elif self.c_type == 'dvine':
            vine = Vine(self.c_type, self.u_matrix, self.tau_mat, self.ppfs, self.y_ind)
            self.model = vine
            self.param = vine.vine_model
        else:
            raise Exception('Unsupported model type: ' + str(self.c_type))

    def infer(self, test_path):
        """predict with the copula model
        Args:
            test_data:path for the test_data file
        """
        # apply cdf transformation to the test data
        test_data = pd.read_csv(
            test_path, sep=',', index_col=False,
            na_values=['NaN', 'nan', 'NULL', 'null'], low_memory=False)
        cdfs, ppfs, u_test = self._preprocessing(test_data)
        if 'vine' in self.c_type:
            vhat = self.model._max_likelihood(u_test)
            LOGGER.debug(vhat)
        return vhat

    def sampling(self, n, plot=False):
        sampled = np.zeros([n, self.n_var])
        if 'vine' in self.c_type:
            for i in range(n):
                sampled[i, :] = self.model._sampling(n)
        if plot:
            plt.scatter(sampled[:, 0], sampled[:, 1], c='red')
            LOGGER.debug(sampled)
            plt.scatter(self.model_data.ix[:, 0], self.model_data.ix[:, 1], c='green')
            plt.show()


class Vine(object):
    """This class constructs a vine model consisting multiple levels of trees
    Attributes:
        c_type: a string indicating the type of the vine model,can be 'cvine' or 'dvine'
        u_matrix: matrix represents the univariate distribution of size m*n,
                where m is the number of data points, and n is number of variables
        tau_mat: matrix represents kendall tau matrix

        y_ind : index of the variable that will be used as y variable
        n_sample: number of samples in the dataset
        n_var: number of variables
        depth: depth of the vine model
        #TO DO: add pruning ?


        vine_model: array [level of tree] -> [tree]
    """

    def __init__(self, c_type, u_mat, tau_mat, ppf, y_ind):

        self.c_type = c_type
        self.u_matrix = u_mat
        self.tau_mat = tau_mat

        self.y_ind = y_ind
        self.n_sample = self.u_matrix.shape[0]
        self.n_var = self.u_matrix.shape[1]
        self.depth = self.n_var - 1
        self.c_map = {0: 'clayton', 1: 'frank', 2: 'gumbel'}

        self.vine_model = []
        self.ppfs = ppf
        self.train_vine()

    def train_vine(self):
        """Train a vine model
        output: trees are stored in self.vine_model
        """
        tree_1 = Tree(0, self)
        self.vine_model.append(tree_1)

        LOGGER.debug('finish building tree : 0')
        LOGGER.debug(str(tree_1))
        self.u_matrix = tree_1.new_U

        for k in range(1, self.depth):
            '''get constraints from previous tree'''
            ctr = self._get_constraints(self.vine_model[k - 1])
            tau = self._get_tau(self.vine_model[k - 1], ctr)
            self.tau_mat = tau
            tree_k = Tree(k, self)
            self.vine_model.append(tree_k)

            LOGGER.debug('finish building tree: %d', k)
            LOGGER.debug(str(tree_k))

            self.u_matrix = tree_k.new_U

    def _get_constraints(self, tree):
        """Get constraints for next tree, where constraints for each node storing its neighboring nodes
        :param tree: a tree instance
        """
        tree_data = tree.tree_data
        constraints = []
        for k in range(tree.n_nodes - 1):
            const_k = []
            for i in range(tree.n_nodes - 1):
                if k != i:
                    # add to constriants if i shared an edge with k
                    if (
                        tree_data[k, 1] == tree_data[i, 1] or
                        tree_data[k, 2] == tree_data[i, 2] or
                        tree_data[k, 1] == tree_data[i, 2] or
                        tree_data[k, 2] == tree_data[i, 1]
                    ):
                        const_k.append(i)
            constraints.append(const_k)
        return constraints

    def _get_tau(self, tree, ctr):
        """Get tau matrix for adjacent pairs
        :param tree: a tree instance
        :param ctr: map of node->adjacent nodes
        """
        tree_data = tree.tree_data
        tau = np.empty([len(ctr), len(ctr)])
        for i in range(len(ctr)):
            links = ctr[i]
            for j in range(len(links)):
                ed1, ed2, ing = tree.identify_eds_ing(tree_data[i, 1:3], tree_data[links[j], 1:3])
                tau[i, links[j]], pvalue = scipy.stats.kendalltau(
                    self.u_matrix[ed1, ing], self.u_matrix[ed2, ing])
        return tau

    def _get_likelihood(self, V, U):
        """Compute total likelihood of the vine model
        """
        U = np.append(U, V)
        values = np.empty([1, len(self.vine_model)])
        for i in range(len(self.vine_model)):
            tree = self.vine_model[i]
            newU, value = tree._likehood_T(U)
            U = newU
            values[0, i] = value
        likelihood = -1.0 * np.sum(values)
        return likelihood

    def _max_likelihood(self, u_matrix):
        LOGGER.debug('Maximize likelihood')
        vhat = np.empty([1, u_matrix.shape[0]])
        for k in range(u_matrix.shape[0]):
            vhat[0, k] = scipy.optimize.fminbound(
                self._get_likelihood, 0, 1, args=(u_matrix[k, :],), xtol=1e-3)
        return vhat

    def _get_adjacent_matrix(self):
        """Build adjacency matrix from the first tree
        """
        first_tree = self.vine_model[0].tree_data
        n = first_tree.shape[0] + 1
        adj = np.zeros([n, n])
        for k in range(n - 1):
            adj[int(first_tree[k, 1]), int(first_tree[k, 2])] = 1
            adj[int(first_tree[k, 2]), int(first_tree[k, 1])] = 1
        return adj

    def _sampling(self, n):
        first_tree = self.vine_model[0].tree_data
        """generating samples from vine model"""
        unis = np.random.uniform(0, 1, self.n_var)
        LOGGER.debug(first_tree)
        # randomly select a node to start with
        first_ind = randint(0, self.n_var - 1)
        adj = self._get_adjacent_matrix()
        visited = []
        explore = []
        explore.insert(0, first_ind)
        itr = 0
        sampled = []
        while explore:
            current = explore.pop(0)
            LOGGER.debug('processing variable : %d', current)
            neighbors = np.where(adj[current, :] == 1)[0].tolist()

            # LOGGER.debug(x)
            # ppfs = self.ppfs[current]
            new_x = self.ppfs[current](unis[current])
            for i in range(itr):
                # get index of edge to retrieve
                index = current or self.vine_model[i].tree_data[:, 2]
                current_ind = np.where(self.vine_model[i].tree_data[:, 1] == index)[0].tolist()[0]
                copula_type = self.vine_model[i].tree_data[current_ind, 4]
                copula_para = self.vine_model[i].tree_data[current_ind, 5]
                LOGGER.debug(copula_type)
                LOGGER.debug(copula_para)
                cop = copulas.Copula(1, 1, theta=-1.1362, cname=self.c_map[copula_type], dev=True)
                tmp = optimize.brentq(
                    cop.derivative,
                    -1000.0,
                    1000.0,
                    args=(sampled[-1], -1.1362, unis[current]))
                LOGGER.debug(tmp)
                # new_x = cop.ppf(unis[itr],u_mat[current,last[-1]],copula_para)
                # tmp = cop.ppf(unis[current],sampled[-1],copula_para)
                # tmp = cop.ppf(unis[current],unis[visited[0]],copula_para)
                new_x = self.ppfs[current](tmp)
                # ppfs = ppfs*cop.ppf
                # LOGGER.debug(new_x)
                # x = x*new_x
            # x = ppfs(u[current])
            # x = self.ppfs[current](new_x)
            sampled.append(new_x)
            for s in neighbors:
                if s in visited:
                    continue
                else:
                    explore.insert(0, s)
            itr += 1
            visited.insert(0, current)
        return sampled


class Tree():
    """instantiate a single tree in the vine model
    :param k: level of tree
    :param prev_T: tree model of previous level
    :param tree_data: current tree model
    :param new_U: conditional cdfs for next level tree
    """

    def __init__(self, k, vine):
        # super(Tree,self).__init__(copula, y_ind)
        self.level = k + 1
        self.vine = vine

        # For each node, tree stores:
        # position k, node index at k, node index at k+1,tau at k,tau_mat at k, tau_mat at k+1
        if self.level == 1:
            self.n_nodes = self.vine.n_var
            self.tree_data = np.empty([self.n_nodes - 1, 6])
            self._build_first_tree()

        else:
            self.prev_T = self.vine.vine_model[k - 1]

            self.n_nodes = self.prev_T.n_nodes - 1
            self.tree_data = np.empty([self.n_nodes - 1, 9])
            self._build_kth_tree()

        # LOGGER.debug(str(self))
        self.new_U = self._data4next_T(self.tree_data)

    def __str__(self):
        """
        print a instance of tree
        """
        if self.vine.c_type == "dvine":
            first = []
            tree_1 = self.vine.vine_model[0].tree_data
            for k in range(tree_1.shape[0]):
                first.append(str(int(tree_1[k, 1])))
            first.append(str(int(tree_1[-1, 2])))
            s = ''
            if self.level == 1:
                # s =''.join(first)
                s = '---'.join(first)
            elif self.level == 2:
                for k in range(self.tree_data.shape[0] + 1):
                    s += first[k] + ',' + first[k + self.level - 1] + '---'
                s = s[:-3]

            else:
                for k in range(self.tree_data.shape[0] + 1):
                    s += first[k] + ',' + first[k + self.level - 1] + '|'
                    for i in range(self.level - 2):
                        s += first[k + i + 1]
                    s += '---'
                s = s[:-3]
        elif self.vine.c_type == "cvine":
            LOGGER.debug('anchor node is :%d', int(self.vine.y_ind))
            s = ''
        return s

    def identify_eds_ing(self, e1, e2):
        """find nodes connecting adjacent edges
        :param e1: pair of nodes representing edge1
        :param e2: pair of nodes representing edge2
        :output ing: nodes connecting e1 and e2
        :output n1,n2: the other node of e1 and e2 respectively
        """
        if e1[0] == e2[0]:
            ing, n1, n2 = e1[0], e1[1], e2[1]
        elif e1[0] == e2[1]:
            ing, n1, n2 = e1[0], e1[1], e2[0]
        elif e1[1] == e2[0]:
            ing, n1, n2 = e1[1], e1[0], e2[1]
        elif e1[1] == e2[1]:
            ing, n1, n2 = e1[1], e1[0], e2[0]
        return int(n1), int(n2), int(ing)

    def _build_first_tree(self):
        """build the first tree with n-1 variable
        """
        # find the pair of maximum tau
        tau_mat = self.vine.tau_mat
        np.fill_diagonal(tau_mat, np.NaN)
        tau_y = tau_mat[:, self.vine.y_ind]
        temp = np.empty([self.n_nodes, 3])
        temp[:, 0] = np.arange(self.n_nodes)
        temp[:, 1] = tau_y
        temp[:, 2] = abs(tau_y)
        temp[np.isnan(temp)] = -10
        tau_sorted = temp[temp[:, 2].argsort()[::-1]]
        if self.vine.c_type == 'cvine':
            # for T1, the anchor node is Y
            self.tree_data[:, 0] = np.arange(self.n_nodes - 1)
            self.tree_data[:, 1] = self.vine.y_ind

            # remove the last row as it is not necessary
            self.tree_data[:, 2] = np.delete(tau_sorted[:, 0], -1, 0)
            self.tree_data[:, 3] = np.delete(tau_sorted[:, 1], -1, 0)
            for k in range(self.n_nodes - 1):
                cop = copulas.Copula(
                    self.vine.u_matrix[:, self.vine.y_ind],
                    self.vine.u_matrix[:, int(self.tree_data[k, 2])],
                    self.tree_data[k, 3]
                )
                self.tree_data[k, 4], self.tree_data[k, 5] = cop.select_copula(cop.U, cop.V)

        if self.vine.c_type == 'dvine':

            left_ind = tau_sorted[0, 0]
            right_ind = tau_sorted[1, 0]
            T1 = np.array([left_ind, self.vine.y_ind, right_ind]).astype(int)
            tau_T1 = tau_sorted[:2, 1]
            # replace tau matrix of the selected variables as a negative number
            # (can't be selected again)
            tau_mat[:, [T1]] = -10

            # greedily build the rest of the first tree
            for k in range(2, self.n_nodes - 1):
                valL, left = np.max(tau_mat[T1[0], :]), np.argmax(tau_mat[T1[0], :])
                valR, right = np.max(tau_mat[T1[-1], :]), np.argmax(tau_mat[T1[-1], :])
                if valL > valR:
                    '''add nodes to the left'''
                    T1 = np.append(int(left), T1)
                    tau_T1 = np.append(valL, tau_T1)
                    tau_mat[:, left] = -10
                else:
                    '''add node to the right'''
                    T1 = np.append(T1, int(right))
                    tau_T1 = np.append(tau_T1, valR)
                    tau_mat[:, right] = -10
            for k in range(self.n_nodes - 1):
                self.tree_data[k, 0] = k
                self.tree_data[k, 1], self.tree_data[k, 2] = T1[k], T1[k + 1]
                self.tree_data[k, 3] = tau_T1[k]

                # Select copula function based on upper and lower tail functions'''

                cop = copulas.Copula(
                    self.vine.u_matrix[:, T1[k]], self.vine.u_matrix[:, T1[k + 1]])
                self.tree_data[k, 4], self.tree_data[k, 5] = cop.select_copula(cop.U, cop.V)

    def _build_kth_tree(self):
        """build tree for level k
        """
        if self.vine.c_type == 'cvine':
            # find anchor variable which has the highest sum of dependence with the rest
            temp = np.empty([self.n_nodes, 2])
            temp[:, 0] = np.arange(self.n_nodes, dtype=int)
            temp[:, 1] = np.sum(abs(self.vine.tau_mat), 1)
            tau_sorted = temp[temp[:, 1].argsort()[::-1]]
            anchor = int(temp[0, 0])
            self.vine.tau_mat[anchor, :] = np.NaN

            # sort the rest of variables based on their dependence with anchor variable
            aux = np.empty([self.n_nodes, 3])
            aux[:, 0] = np.arange(self.n_nodes, dtype=int)
            aux[:, 1] = self.vine.tau_mat[:, anchor]
            aux[:, 2] = abs(self.vine.tau_mat[:, anchor])
            aux[anchor, 2] = -10
            self.tree_data[:, 0] = np.arange(self.n_nodes - 1)
            self.tree_data[:, 1] = anchor
            self.tree_data[:, 2] = np.delete(tau_sorted[:, 0], -1, 0)
            self.tree_data[:, 3] = np.delete(tau_sorted[:, 1], -1, 0)

        if self.vine.c_type == 'dvine':
            for k in range(self.n_nodes - 1):
                self.tree_data[k, 0] = k
                self.tree_data[k, 1], self.tree_data[k, 2] = int(k), int(k + 1)
                self.tree_data[k, 3] = self.vine.tau_mat[k, k + 1]

        """select copula function"""
        for k in range(self.n_nodes - 1):
            [ed1, ed2, ing] = self.identify_eds_ing(
                self.prev_T.tree_data[k, 1:3], self.prev_T.tree_data[k + 1, 1:3])
            U1 = self.vine.u_matrix[ed1, ing]
            U2 = self.vine.u_matrix[ed2, ing]
            cop = copulas.Copula(U1, U2, self.tree_data[k, 3])
            self.tree_data[k, 4], self.tree_data[k, 5] = cop.select_copula(cop.U, cop.V)
            self.tree_data[k, 6], self.tree_data[k, 7] = ed1, ed2
            self.tree_data[k, 8] = ing

    def _data4next_T(self, tree):
        """
        prepare conditional tau matrix for next tree
        """
        eps = np.finfo(np.float32).eps
        n = self.vine.n_var
        U = np.empty([n, n], dtype=object)
        # U = self.vine.u_matrix
        for k in range(tree.shape[0]):
            copula_name = self.vine.c_map[int(tree[k, 4])]
            copula_para = tree[k, 5]
            if self.level == 1:
                U1 = self.vine.u_matrix[:, int(tree[k, 1])]
                U2 = self.vine.u_matrix[:, int(tree[k, 2])]
            else:
                U1 = self.vine.u_matrix[int(tree[k, 6]), int(tree[k, 8])]
                U2 = self.vine.u_matrix[int(tree[k, 7]), int(tree[k, 8])]

            '''compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/dui'''
            U1 = [x for x in U1 if x is not None]
            U2 = [x for x in U2 if x is not None]

            c1 = copulas.Copula(U2, U1, theta=copula_para, cname=copula_name, dev=True)
            U1givenU2 = c1.derivative(U2, U1, copula_para)
            U2givenU1 = c1.derivative(U1, U2, copula_para)

            '''correction of 0 or 1'''
            U1givenU2[U1givenU2 == 0], U2givenU1[U2givenU1 == 0] = eps, eps
            U1givenU2[U1givenU2 == 1], U2givenU1[U2givenU1 == 1] = 1 - eps, 1 - eps

            U[int(tree[k, 1]), int(tree[k, 2])] = U1givenU2
            U[int(tree[k, 2]), int(tree[k, 1])] = U2givenU1
        return U

    def _likehood_T(self, U):
        """Compute likelihood of the tree given an U matrix"""
        newU = np.empty([self.vine.n_var, self.vine.n_var])
        tree = self.tree_data
        values = np.zeros([1, tree.shape[0]])
        for i in range(tree.shape[0]):
            cname = self.vine.c_map[int(tree[i, 4])]
            v1 = int(tree[i, 1])
            v2 = int(tree[i, 2])
            copula_para = tree[i, 5]
            if self.level == 1:
                U_arr = np.array([U[v1]])
                V_arr = np.array([U[v2]])
                cop = copulas.Copula(U_arr, V_arr, theta=copula_para, cname=cname, dev=True)
                values[0, i] = cop.pdf(U_arr, V_arr, copula_para)
                U1givenU2 = cop.derivative(V_arr, U_arr, copula_para)
                U2givenU1 = cop.derivative(U_arr, V_arr, copula_para)
            else:
                v1 = int(tree[i, 6])
                v2 = int(tree[i, 7])
                joint = int(tree[i, 8])
                U1 = np.array([U[v1, joint]])
                U2 = np.array([U[v2, joint]])
                cop = copulas.Copula(U1, U2, theta=copula_para, cname=cname, dev=True)
                values[0, i] = cop.pdf(U1, U2, theta=copula_para)
                U1givenU2 = cop.derivative(U2, U1, copula_para)
                U2givenU1 = cop.derivative(U1, U2, copula_para)
            newU[v1, v2] = U1givenU2
            newU[v2, v1] = U2givenU1
        # LOGGER.debug(values)
        value = np.sum(np.log(values))
        return newU, value
