import numpy as np
import scipy

from copulas.bivariate.copulas import Copula

c_map = {0: 'clayton', 1: 'frank', 2: 'gumbel'}
eps = np.finfo(np.float32).eps


class Tree(object):
    """Helper class to instantiate a single tree in the vine model
    """

    def __init__(self, k, n, tau_mat, prev_T):
        self.level = k + 1
        self.prev_T = prev_T
        self.n_nodes = n
        self.edge_set = []
        self.tau_mat = tau_mat
        if self.level == 1:
            self.u_matrix = prev_T
            self._build_first_tree()
        else:
            self._build_kth_tree()
            self._data4next_T()

    def _identify_eds_ing(self, e1, e2):
        """find nodes connecting adjacent edges
        :param e1: pair of nodes representing edge1
        :param e2: pair of nodes representing edge2
        :return left,right: end node indices of the new edge
        :return D: dependence_set of the new edge
        """
        A = set([e1.L, e1.R])
        A.update(e1.D)
        B = set([e2.L, e2.R])
        B.update(e2.D)
        D = list(A & B)
        left = list(A ^ B)[0]
        right = list(A ^ B)[1]
        return left, right, D

    def _check_adjacency(self, e1, e2):
        """check if two edges are adjacent"""
        return (e1.L == e2.L or e1.L == e2.R or e1.R == e2.L or e1.R == e2.R)

    def _check_contraint(self, e1, e2):
        full_node = set([e1.L, e1.R, e2.L, e2.R])
        full_node.update(e1.D)
        full_node.update(e2.D)
        return (len(full_node) == (self.level + 1))

    def _get_constraints(self):
        """get neighboring edges
        """
        for k in range(len(self.edge_set)):
            for i in range(len(self.edge_set)):
                # add to constriants if i shared an edge with k
                if k != i:
                    if self._check_adjacency(self.edge_set[k], self.edge_set[i]):
                        self.edge_set[k].neighbors.append(i)

    def _get_tau(self):
        """Get tau matrix for adjacent pairs
        :param tree: a tree instance
        :param ctr: map of edge->adjacent edges
        """
        tau = np.empty([len(self.edge_set), len(self.edge_set)])
        for i in range(len(self.edge_set)):
            for j in self.edge_set[i].neighbors:
                edge = self.edge_set[i].parent
                l_p = edge[0]
                r_p = edge[1]
                if self.level == 1:
                    U1, U2 = self.u_matrix[:, l_p], self.u_matrix[:, r_p]
                else:
                    U1, U2 = self.prev_T.edge_set[l_p].U,
                    self.prev_T.edge_set[r_p].U
                tau[i, j], pvalue = scipy.stats.kendalltau(U1, U2)
        return tau

    def _data4next_T(self):
        """
        prepare conditional U matrix for next tree
        """
        # U = np.empty([self.n_nodes,self.n_nodes],dtype=object)
        edge_set = self.edge_set
        for k in range(len(edge_set)):
            edge = edge_set[k]
            copula_name = c_map[edge.name]
            copula_para = edge.param
            if self.level == 1:
                U1, U2 = self.u_matrix[:, edge.L], self.u_matrix[:, edge.R]
            else:
                prev_T = self.prev_T.edge_set
                l_p = edge.parent[0]
                r_p = edge.parent[1]
                U1, U2 = prev_T[l_p].U, prev_T[r_p].U
            # compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/dui'''
            U1 = [x for x in U1 if x is not None]
            U2 = [x for x in U2 if x is not None]
            c1 = Copula(cname=copula_name)
            c1.fit(U1, U2)
            derivative = c1.get_h_function()
            U1givenU2 = derivative(U2, U1, copula_para)
            U2givenU1 = derivative(U1, U2, copula_para)
            # correction of 0 or 1'''
            U1givenU2[U1givenU2 == 0], U2givenU1[U2givenU1 == 0] = eps, eps
            U1givenU2[U1givenU2 == 1], U2givenU1[U2givenU1 == 1] = 1 - eps, 1 - eps
            edge.U = U1givenU2

    def _likehood_T(self, U):
        """Compute likelihood of the tree given an U matrix
        """
        newU = np.empty([self.vine.n_var, self.vine.n_var])
        edge_set = self.edge_set
        values = np.zeros([1, len(edge_set)])
        for i in range(len(edge_set)):
            edge = edge_set[i]
            cname = self.vine.c_map[edge.name]
            v1 = edge.L
            v2 = edge.R
            copula_para = edge.param
            if self.level == 1:
                U1 = np.array([U[v1]])
                U2 = np.array([U[v2]])
            else:
                prev_T = self.prev_T.edge_set
                l_p = edge.parent[0]
                r_p = edge.parent[1]
                U1, U2 = prev_T[l_p].U, prev_T[r_p].U
            cop = Copula(cname=cname)
            cop.fit(U1, U2)
            pdf = cop.get_pdf()
            derivative = cop.get_h_function()
            values[0, i] = pdf(U1, U2)
            U1givenU2 = derivative(U2, U1, copula_para)
            # U2givenU1 = derivative(U1, U2, copula_para)
            edge.U = U1givenU2
            value = np.sum(np.log(values))
        return newU, value

    def print_tree(self):
        for e in self.edge_set:
            print(e.L, e.R, e.D, e.parent)


class CTree(Tree):
    def __init__(self, k, n, tau_mat, prev_T):
        super(CTree, self).__init__(k, n, tau_mat, prev_T)

    def _build_first_tree(self):
        # first column is the variable of interest
        tau_mat = self.tau_mat
        np.fill_diagonal(tau_mat, np.NaN)
        tau_y = tau_mat[:, 0]
        temp = np.empty([self.n_nodes, 3])
        temp[:, 0] = np.arange(self.n_nodes)
        temp[:, 1] = tau_y
        temp[:, 2] = abs(tau_y)
        temp[np.isnan(temp)] = -10
        tau_sorted = temp[temp[:, 2].argsort()[::-1]]
        for itr in range(1, self.n_nodes):
            ind = tau_sorted[itr, 0]
            name, param = Copula.select_copula(self.u_matrix[:, 0],
                                               self.u_matrix[:, ind])
            new_edge = Edge(itr, 0, ind, tau_mat[0, ind], name, param)
            new_edge.parent.append(0)
            new_edge.parent.append(ind)
            self.edge_set.append(new_edge)

    def _build_kth_tree(self):
        # find anchor variable with highest sum of dependence with the rest
        temp = np.empty([self.n_nodes, 2])
        temp[:, 0] = np.arange(self.n_nodes, dtype=int)
        temp[:, 1] = np.sum(abs(self.tau_mat), 1)
        tau_sorted = temp[temp[:, 1].argsort()[::-1]]
        anchor = int(temp[0, 0])
        self.tau_mat[anchor, :] = np.NaN
        # sort the rest of variables based on dependence with anchor variable
        aux = np.empty([self.n_nodes, 3])
        aux[:, 0] = np.arange(self.n_nodes, dtype=int)
        aux[:, 1] = self.tau_mat[:, anchor]
        aux[:, 2] = abs(self.tau_mat[:, anchor])
        aux[anchor, 2] = -10
        aux_sorted = aux[aux[:, 2].argsort()[::-1]]
        for itr in range(self.n_nodes - 1):
            right = aux_sorted[itr, 0]
            U1, U2 = self.prev_T.edge_set[anchor].U,
            self.prev_T.edge_set[right].U
            name, param = Copula.select_copula(U1, U2)
            [ed1, ed2, ing] =\
                self.identify_eds_ing(self.prev_T.edge_set[anchor],
                                      self.prev_T.edge_set[right])
            new_edge = Edge(itr, ed1, ed2, tau_sorted[itr, 1], name, param)
            new_edge.parent.append(anchor)
            new_edge.parent.append(right)
            new_edge.parents.append(self.prev_T.edge_set[anchor])
            new_edge.parents.append(self.prev_T.edge_set[right])
            self.edge_set.append(new_edge)


class DTree(Tree):
    def __init__(self, k, n, tau_mat, prev_T):
        super(DTree, self).__init__(k, n, tau_mat, prev_T)

    def _build_first_tree(self):
        # find the pair of maximum tau
        tau_mat = self.tau_mat
        np.fill_diagonal(tau_mat, np.NaN)
        tau_y = tau_mat[:, 0]
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
        tau_mat[:, [T1]] = -10
        for k in range(2, self.n_nodes - 1):
            valL, left = np.max(tau_mat[T1[0], :]),
            np.argmax(tau_mat[T1[0], :])
            valR, right = np.max(tau_mat[T1[-1], :]),
            np.argmax(tau_mat[T1[-1], :])
            if valL > valR:
                # add nodes to the left'''
                T1 = np.append(int(left), T1)
                tau_T1 = np.append(valL, tau_T1)
                tau_mat[:, left] = -10
            else:
                # add node to the right'''
                T1 = np.append(T1, int(right))
                tau_T1 = np.append(tau_T1, valR)
                tau_mat[:, right] = -10
        for k in range(self.n_nodes - 1):
            name, param = Copula.select_copula(self.u_matrix[:, T1[k]],
                                               self.u_matrix[:, T1[k + 1]])
            new_edge = Edge(k, T1[k], T1[k + 1], tau_T1[k], name, param)
            new_edge.parent.append(T1[k])
            new_edge.parent.append(T1[k + 1])
            self.edge_set.append(new_edge)

    def _build_kth_tree(self):
        for k in range(self.n_nodes - 1):
            l_p = self.prev_T.edge_set[k]
            r_p = self.prev_T.edge_set[k + 1]
            name, param = Copula.select_copula(l_p.U, r_p.U)
            [ed1, ed2, ing] = self.identify_eds_ing(l_p, r_p)
            new_edge = Edge(k, ed1, ed2, self.tau_mat[k, k + 1], name, param)
            self.edge_set.append(new_edge)


class RTree(Tree):
    def __init__(self, k, n, tau_mat, prev_T):
        super(RTree, self).__init__(k, n, tau_mat, prev_T)

    def _build_first_tree(self):
        """build the first t ree with n-1 variable
        """
        tau_mat = self.tau_mat
        # Prim's algorithm
        neg_tau = -1.0 * abs(tau_mat)
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
            name, param = Copula.select_copula(self.u_matrix[:, edge[0]],
                                               self.u_matrix[:, edge[1]])
            new_edge = Edge(itr, edge[0], edge[1], tau_mat[edge[0], edge[1]],
                            name, param)
            new_edge.parent.append(edge[0])
            new_edge.parent.append(edge[1])
            self.edge_set.append(new_edge)
            X.add(edge[1])
            itr += 1

    def _build_kth_tree(self):
        """build tree for level k
        """
        neg_tau = -abs(self.tau_mat)
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
                        if self.check_contraint(self.prev_T.edge_set[x],
                                                self.prev_T.edge_set[k]):
                            adj_set.add((x, k))
            # find edge with maximum tau
            # print('processing edge:{0}'.format(x))
            if len(list(adj_set)) == 0:
                visited.add(list(unvisited)[0])
                continue
            edge = sorted(adj_set, key=lambda e: neg_tau[e[0]][e[1]])[0]
            [ed1, ed2, ing] =\
                self.identify_eds_ing(self.prev_T.edge_set[edge[0]],
                                      self.prev_T.edge_set[edge[1]])
            # U1 = self.u_matrix[ed1,ing]
            # U2 = self.u_matrix[ed2,ing]
            l_p = edge[0]
            r_p = edge[1]
            U1, U2 = self.prev_T.edge_set[l_p].U, self.prev_T.edge_set[r_p].U
            name, param = Copula.select_copula(U1, U2)
            new_edge = Edge(itr, ed1, ed2, self.tau_mat[edge[0], edge[1]],
                            name, param)
            new_edge.D = ing
            new_edge.parent.append(edge[0])
            new_edge.parent.append(edge[1])
            new_edge.parents.append(self.prev_T.edge_set[edge[0]])
            new_edge.parents.append(self.prev_T.edge_set[edge[1]])
            # new_edge.likelihood = np.log(cop.pdf(U1,U2,param))
            self.edge_set.append(new_edge)
            visited.add(edge[1])
            unvisited.remove(edge[1])
            itr += 1


class Edge(object):
    def __init__(self, index, left, right, tau, copula_name, copula_para):
        self.index = index  # index of the edge in the current tree
        self.level = None   # in which level of tree
        self.L = left   # left_node index
        self.R = right  # right_node index
        self.D = []  # dependence_set
        self.parent = []  # indices of parent edges in the previous tree
        self.parents = []
        self.tau = tau   # correlation of the edge
        self.name = copula_name
        self.param = copula_para
        self.U = None
        self.likelihood = None
        self.neighbors = []

    def get_likehood(self, U):
        """Compute likelihood given a U matrix
        """
        if self.level == 1:
            name, param = Copula.select_copula(U[:, self.L], U[:, self.R])
            cop = Copula(cname=name)
            cop.fit(U[:, self.L], U[:, self.R])
            pdf = cop.get_pdf()
            self.likelihood = pdf(U[:, self.L], U[:, self.R])
        else:
            l_p = self.parents[0]
            r_p = self.parents[1]
            U1, U2 = l_p.U, r_p.U
            name, param = Copula.select_copula(U1, U2)
            cop = Copula(cname=name)
            cop.fit(U1, U2)
            pdf = cop.get_pdf()
            self.likelihood = pdf(U1, U2)
