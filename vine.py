import numpy as np
import scipy 

import copula

class Vine(object):

    def __init__(self, copula, y_ind):
        """Represents a vine model consisting multiple levels of trees
        param ctype: the type of copula models, can be 'Gaussian','cvine','dvine'
        param u_matrix: matrix represents the univariate distribution of size m*n, where m is the number of data points, and n is number of variables
        param n_var: number of variables
        param depth: depth(number of trees) of the vine model
        param tau_mat: the kendall tau matrix
        param y_ind: index of the variable that will be used as y variable
        param dat_V: univariate cdf of the output variables
        param dat_U: univariate cdf of the input variables
        param vine_model: array [level of tree] -> [tree]
        """
        # self.copula = copula
        self.ctype = copula.ctype
        self.u_matrix = copula.unis
        self.n_var = copula.n_var
        print 'number of variables: %d'%(self.n_var)
        self.depth = self.n_var - 1
        self.tau_mat = copula.param
        if y_ind is None:
            self.y_ind = self.n_var - 1
        else:
            self.y_ind = y_ind
        self.dat_V = self.u_matrix[:,self.y_ind]     
        self.dat_U = np.delete(self.u_matrix, self.y_ind, axis=1) 
        self.vine_model=[]

    def train_vine(self):
        """Train a vine model
        output: trees are stored in self.vine_model
        """
        tree_1 = Tree(0,self)
        print('finish building tree : 0')
        self.vine_model.append(tree_1)
        self.u_matrix = tree_1.new_U

        for k in range(1,self.depth):
            '''get constraints from previous tree'''
            ctr = self._get_constraints(self.vine_model[k-1])
            tau = self._get_tau(self.vine_model[k-1],ctr)
            self.tau_mat = tau
            tree_k = Tree(k,self)
            self.vine_model.append(tree_k)
            self.u_matrix = tree_k.new_U
            print'finish building tree: %d'%(k)




    def _get_constraints(self,tree):
        """


        """
        tree_data = tree.tree_data
        constraints = []
        for k in range(tree.n_nodes-1):
            const_k = []
            for i in range(tree.n_nodes-1):
                if k!=i:
                    if tree_data[k,1]==tree_data[i,1] or tree_data[k,2]==tree_data[i,2] or tree_data[k,1]==tree_data[i,2] or tree_data[k,2]==tree_data[i,1]:
                        const_k.append(i)
            constraints.append(const_k)
        return constraints

    def _get_tau(self,tree,ctr):
        tree_data = tree.tree_data
        tau = np.empty([len(ctr),len(ctr)])
        for i in range(len(ctr)):
            links = ctr[i]
            for j in range(len(links)):
                ed1,ed2,ing = tree.identify_eds_ing(tree_data[i,1:3],tree_data[links[j],1:3])
                tau[i,links[j]],pvalue = scipy.stats.kendalltau(self.u_matrix[ed1,ing],self.u_matrix[ed2,ing])
        return tau

    def predict(self,test_data):
        """compute univariate distribution for the test data"""
        copula = copula.CopulaUtil()



class Tree():
    """instantiate a single tree in the vine model
    :param k: level of tree
    :param prev_T: tree model of previous level
    :param tree_data: current tree model
    :param new_U: conditional cdfs for next level tree
    """

    def __init__(self, k, vine):
        # super(Tree,self).__init__(copula, y_ind)
        self.level = k+1
        self.vine = vine
       
            
        '''For each node, tree stores position k, node index at k, node index at k+1,tau at k,tau_mat at k, tau_mat at k+1'''
        if self.level ==1 :
            self.n_nodes = self.vine.n_var
            self.tree_data = np.empty([self.n_nodes-1,6])
            self._build_first_tree()
            self.print_tree()

        else:
            self.prev_T = self.vine.vine_model[k-1]

            self.n_nodes = self.prev_T.n_nodes-1
            self.tree_data = np.empty([self.n_nodes-1,9])
            self._build_kth_tree()
            
        # self.print_tree()
        self.new_U = self._data4next_T(self.tree_data)


    def identify_eds_ing(self,e1,e2):
        if e1[0]== e2[0]:
            ing,e1,e2 = e1[0],e1[1],e2[1]
        elif e1[0]== e2[1]:
            ing,e1,e2 = e1[0],e1[1],e2[0]
        elif e1[1]==e2[0]:
            ing,e1,e2 = e1[1],e1[0],e2[1]
        elif e1[1]==e2[1]:
            ing,e1,e2 = e1[1],e1[0],e2[0]
        return int(e1),int(e2),int(ing)




    def _build_first_tree(self):
        """build the first tree with n-1 variable"""
        """find the pair of maximum tau"""
        tau_mat = self.vine.tau_mat
        np.fill_diagonal(tau_mat,np.NaN)
        tau_y = tau_mat[:,self.vine.y_ind]
        N = len(tau_y)
        temp=np.empty([self.n_nodes,3])
        temp[:,0] = np.arange(self.n_nodes)
        temp[:,1] = tau_y
        temp[:,2] = abs(tau_y)
        temp[np.isnan(temp)] = -10
        tau_sorted = temp[temp[:,2].argsort()[::-1]]
        if self.vine.ctype == 'cvine':
            """for T1, the anchor node is Y"""
            self.tree_data[:,0] = np.arange(self.n_nodes-1)
            self.tree_data[:,1] = self.vine.y_ind
            """remove the last row as it is not necessary"""
            self.tree_data[:,2] = np.delete(tau_sorted[:,0],-1,0)
            self.tree_data[:,3] = np.delete(tau_sorted[:,1],-1,0)
            for k in xrange(self.n_nodes-1):
                self.tree_data[k,4],self.tree_data[k,5]=copula.select_copula(self.vine.u_matrix[:,self.vine.y_ind],self.vine.u_matrix[:,int(self.tree_data[k,2])],self.tree_data[k,3])
            
        if self.vine.ctype == 'dvine':
            
            left_ind = tau_sorted[0,0]
            right_ind = tau_sorted[1,0]
            T1 = np.array([left_ind,self.vine.y_ind,right_ind]).astype(int)
            tau_T1 = tau_sorted[:2,1] 
            '''replace tau matrix of the selected variables as a negative number (can't be selected again)'''
            tau_mat[:,[T1]] = -10
            # print(tau_mat)
            '''greedily build the rest of the first tree'''
            for k in xrange(2,self.n_nodes-1):
                # print(k)
                valL,left=np.max(tau_mat[T1[0],:]),np.argmax(tau_mat[T1[0],:])
                valR,right=np.max(tau_mat[T1[-1],:]),np.argmax(tau_mat[T1[-1],:])
                if valL>valR:
                    '''add nodes to the left'''
                    T1=np.append(int(left),T1)
                    tau_T1=np.append(valL,tau_T1)
                    tau_mat[:,left]= -10
                else:
                    '''add node to the right'''
                    T1=np.append(T1,int(right))
                    tau_T1=np.append(tau_T1,valR)
                    tau_mat[:,right]= -10
            for k in xrange(self.n_nodes-1):
                self.tree_data[k,0]=k
                self.tree_data[k,1],self.tree_data[k,2]=T1[k],T1[k+1]
                self.tree_data[k,3]=tau_T1[k]
                '''Select copula function based on upper and lower tail functions'''
                # print(Copula.select_copula(self.u_matrix[:,T1[k]],self.u_matrix[:,T1[k+1]]))
                self.tree_data[k,4],self.tree_data[k,5]=copula.select_copula(self.vine.u_matrix[:,T1[k]],self.vine.u_matrix[:,T1[k+1]])



    def _build_kth_tree(self):
        """build tree for level k"""
        if self.vine.ctype == 'cvine':
            """find anchor variable which has the highest sum of dependence with the rest"""
            temp=np.empty([self.n_nodes,2])
            temp[:,0] = np.arange(self.n_nodes,dtype=int)
            temp[:,1] = np.sum(abs(self.vine.tau_mat),1)
            tau_sorted = temp[temp[:,1].argsort()[::-1]]
            anchor = int(temp[0,0])
            self.vine.tau_mat[anchor,:] = np.NaN
            """sort the rest of variables based on their dependence with anchor variable """
            aux =np.empty([self.n_nodes,3])
            aux[:,0] = np.arange(self.n_nodes,dtype=int)
            aux[:,1] = self.vine.tau_mat[:,anchor]
            aux[:,2] = abs(self.vine.tau_mat[:,anchor])
            aux[anchor,2] = -10
            aux_sorted = aux[aux[:,2].argsort()[::-1]]
            self.tree_data[:,0] = np.arange(self.n_nodes-1)
            self.tree_data[:,1] = anchor
            self.tree_data[:,2] = np.delete(tau_sorted[:,0],-1,0)
            self.tree_data[:,3] = np.delete(tau_sorted[:,1],-1,0)

        if self.vine.ctype == 'dvine':
            for k in xrange(self.n_nodes-1):
                self.tree_data[k,0]=k
                self.tree_data[k,1],self.tree_data[k,2]=int(k),int(k+1)
                self.tree_data[k,3]=self.vine.tau_mat[k,k+1]

        """select copula function"""
        for k in xrange(self.n_nodes-1):
                [ed1,ed2,ing] = self.identify_eds_ing(self.prev_T.tree_data[k,1:3],self.prev_T.tree_data[k+1,1:3])
                U1 = self.vine.u_matrix[ed1,ing]
                U2 = self.vine.u_matrix[ed2,ing]
                self.tree_data[k,4],self.tree_data[k,5] = copula.select_copula(U1,U2,self.tree_data[k,3])
                self.tree_data[k,6],self.tree_data[k,7] = ed1, ed2
                self.tree_data[k,8] = ing

    
    def _data4next_T(self,tree):
        eps = np.finfo(np.float32).eps
        U = np.empty([tree.shape[0]+1,tree.shape[0]+1],dtype=object)
        # print(tree)

        for k in xrange(tree.shape[0]):
            copula_name = int(tree[k,4])
            copula_para = tree[k,5]
            if self.level == 1:
                U1,U2 = self.vine.u_matrix[:,int(tree[k,1])],self.vine.u_matrix[:,int(tree[k,2])]
            else:
                U1,U2 = self.vine.u_matrix[int(tree[k,6]),int(tree[k,8])],self.vine.u_matrix[int(tree[k,7]),int(tree[k,8])]
            '''compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/dui'''
            U1givenU2 = copula.du_copula(U2,U1,copula_para,copula_name)
            U2givenU1 = copula.du_copula(U1,U2,copula_para,copula_name)
            # print(U1givenU2)

            '''correction of 0 or 1'''
            U1givenU2[U1givenU2==0],U2givenU1[U2givenU1==0]=eps,eps
            U1givenU2[U1givenU2==1],U2givenU1[U2givenU1==1]=1-eps,1-eps

            U[int(tree[k,1]),int(tree[k,2])]=U1givenU2
            U[int(tree[k,2]),int(tree[k,1])]=U2givenU1
        return U

    def print_tree(self):
        if self.vine.ctype == "dvine":
            tree = list(self.tree_data[:,1].astype(int))
            tree.append(int(self.tree_data[-1,2]))
        elif self.vine.ctype == "cvine":
            print"anchor node is :%d"%(int(self.vine.y_ind))
            tree = list(self.tree_data[:,2].astype(int))
        print(tree)




if __name__ == '__main__':
    copula = copula.CopulaUtil('lucas0_train.csv','kde','dvine')
    dvine = Vine(copula,11)
    dvine.train_vine()
    print(dvine.vine_model[-1].tree_data)