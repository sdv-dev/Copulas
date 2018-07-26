from copulas.univariate.KDEUnivariate import KDEUnivariate
from copulas.multivariate.MVCopula import MVCopula
from copulas.multivariate.Tree import Tree, CTree, DTree, RTree


class VineCopula(MVCopula):
    """ Class for a vine copula model """

    def __init__(self):
        super(VineCopula, self).__init__(type)
        self.type = type
        self.u_matrix = None

        self.cdf = None
        self.ppf = None

        self.model = None
        self.param = None

    def fit(self, data):
        """Fit a vine model to the data
        Returns:
        """
        self.data = data
        self.n_sample = self.data.shape[0]
        self.n_var = self.data.shape[1]
        self.tau_mat = self.data.corr(method='kendall').as_matrix()
        self.u_mat = np.empty([self.n_sample, self.n_var])
        self.unis = []
        for col in data:
            uni = KDEUnivariate()
            uni.fit(data)
            self.u_mat[:, col] = [uni.get_cdf(x) for x in col]
            self.unis.append(uni)
        self.truncated = truncated
        self.depth = self.n_var - 1
        self.vine_model = []
        if self.type == 'cvine':
            model = CVine(self.u_mat, self.tau_mat, self.truncated)
        elif self.type == 'dvine':
            model = DVine(self.u_mat, self.tau_mat, self.truncated)
        elif self.type == 'rvine':
            model = RVine(self.u_mat, self.tau_mat, self.truncated)
        else:
            raise Exception('Unsupported vine copula type: ' + str(self.cname))
        model.train_vine()

    def sample(self, num_rows=1):
        first_tree = self.vine_model[0].edge_set
        """generating samples from vine model"""
        unis = np.random.uniform(0, 1, self.n_var)
        # randomly select a node to start with
        first_ind = randint(0, self.n_var-1)
        adj = self._get_adjacent_matrix()
        visited, explore = [], []
        explore.insert(0, first_ind)
        sampled = [0]*self.n_var
        itr = 0
        while explore:
            current = explore.pop(0)
            # print('processing variable : {0}'.format(current))
            neighbors = np.where(adj[current, :] == 1)[0].tolist()
            if itr == 0:
                new_x = self.ppfs[current](unis[current])
            else:
                for i in range(itr-1, -1, -1):
                    current_ind = -1
                    if i >= self.truncated:
                        continue
                    current_tree = self.vine_model[i].edge_set
                    # print('inside loop number: {0}'.format(i))
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
                                if condition.issubset(visited):
                                    current_ind = edge.index
                                break
                    if current_ind != -1:
                        # the node is not indepedent contional on visited node
                        copula_type = current_tree[current_ind].name
                        copula_para = current_tree[current_ind].param
                        cop = copula.Copula(1, 1, theta=copula_para,
                                            cname=c_map[copula_type], dev=True)
                        # start with last level
                        if i == itr - 1:
                            tmp = optimize.fminbound(cop.derivative, eps, 1.0,
                                                     args=(unis[visited[0]],
                                                           copula_para,
                                                           unis[current]))
                        else:
                            tmp = optimize.fminbound(cop.derivative, eps, 1.0,
                                                     args=(unis[visited[0]],
                                                           copula_para, tmp))
                        mp = min(max(tmp, eps), 0.99)
                new_x = self.ppfs[current](tmp)
            # print(new_x)
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
        n = len(first_tree)+1
        adj = np.zeros([n, n])
        for k in range(len(first_tree)):
            adj[first_tree[k].L, first_tree[k].R] = 1
            adj[first_tree[k].R, first_tree[k].L] = 1
        return adj


class RVine(VineCopula):
    def train_vine(self):
        """Train a vine model
        output: trees are stored in self.vine_model
        """
        LOGGER.debug('Fitting RVine Copula')
        print('start building tree : 0')
        tree_1 = RTree(0, self.n_var, self.tau_mat, self.u_matrix)
        self.vine_model.append(tree_1)
        print('finish building tree : 0')
        tree_1.print_tree()
        for k in range(1, min(self.n_var-1, self.truncated)):
            # get constraints from previous tree'''
            self.vine_model[k-1]._get_constraints()
            tau = self.vine_model[k-1]._get_tau()
            print('start building tree: {0}'.format(k))
            tree_k = RTree(k, self.n_var-k, tau, self.vine_model[k-1])
            self.vine_model.append(tree_k)
            print('finish building tree: {0}'.format(k))
            tree_k.print_tree()


class CVine(VineCopula):
        def train_vine(self):
            """Train a vine model
            output: trees are stored in self.vine_model
            """
            LOGGER.debug('Fitting CVine Copula')
            print('start building tree : 0')
            tree_1 = CTree(0, self.n_var, self.tau_mat, self.u_matrix)
            self.vine_model.append(tree_1)
            print('finish building tree : 0')
            tree_1.print_tree()
            for k in range(1, min(self.n_var-1, self.truncated)):
                # get constraints from previous tree'''
                self.vine_model[k-1]._get_constraints()
                tau = self.vine_model[k-1]._get_tau()
                print('start building tree: {0}'.format(k))
                tree_k = CTree(k, self.n_var-k, tau, self.vine_model[k-1])
                self.vine_model.append(tree_k)
                print('finish building tree: {0}'.format(k))
                tree_k.print_tree()


class DVine(VineCopula):
        def train_vine(self):
            """Train a vine model
            output: trees are stored in self.vine_model
            """
            LOGGER.debug('Fitting DVine Copula')
            print('start building tree : 0')
            tree_1 = DTree(0, self.n_var, self.tau_mat, self.u_matrix)
            self.vine_model.append(tree_1)
            print('finish building tree : 0')
            tree_1.print_tree()
            for k in range(1, min(self.n_var-1, self.truncated)):
                '''get constraints from previous tree'''
                self.vine_model[k-1]._get_constraints()
                tau = self.vine_model[k-1]._get_tau()
                print('start building tree: {0}'.format(k))
                tree_k = DTree(k, self.n_var-k, tau, self.vine_model[k-1])
                self.vine_model.append(tree_k)
                print('finish building tree: {0}'.format(k))
                tree_k.print_tree()


if __name__ == '__main__':
    print('start building')
    data = pd.read_csv('../../data/iris.data.csv')
    model = VineCopula()
    model.fit(data)
    print(model)
    # sample=model.sampling(400,plot=False,out_dir='../experiments/cancer_ri_synthetic.csv')
