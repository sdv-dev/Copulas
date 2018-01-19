import scipy

# import copulalib
import numpy as np
import pandas as pd

import utils



class CopulaException(Exception):
    pass

class Copula(object):
    def __init__(self, full_path , utype, ctype):
        """Instantiates an instance of the copula object 

        :param utype: the distribution for the univariate, can be 'kde','norm'
        :param ctype: the type of copula models, can be 'Gaussian','cvine','dvine'
        :param cname: the choice of copulas, can be 'clayton','gumbel'  

        """
        self.utype = utype
        self.ctype = ctype
        # self.cname = cname
        self.args = None
        self.corr_measure = 'Pearson'
        self.model_data = pd.read_csv(full_path, sep=',', index_col=False, 
            na_values=['NaN', 'nan', 'NULL', 'null'], low_memory=False)
        self.n_sample = self.model_data.shape[0]
        self.n_var = self.model_data.shape[1]
        self.cdfs,self.unis= self._train_cdf()
        self._fit_copula()


    def _train_cdf(self):
        """Find the cdf for each columns.
        :output cdfs: list of cdf function for each variables
        :output unis: np matrix of data after applying cdf function to each column
        """
        cdfs = []
        unis = np.empty([self.n_sample,self.n_var])
        count = 0
        for col in self.model_data:
            dist = utils.Distribution(column=self.model_data[col].values)
            dist.name=self.utype
            cdf = dist.cdf
            cdfs.append(cdf)
            unis[:,count]=[cdf(x) for x in list(self.model_data[col].values)]
            count+=1
        return cdfs,unis

    def _fit_copula(self):
        """Fits a copula to the data 
        :output self.param: linear correlation matrix for Gaussian, Kendall's tau for others
        """
        if self.ctype == 'Gaussian':
            self.param = self.model_data.corr(method='pearson').as_matrix()
        else:
            self.param = self.model_data.corr(method='kendall').as_matrix()

        

    def density_gaussian(self,u):
        """Compute density of gaussian copula"""
        R = cholesky(self.param)
        x = norm.ppf(u)
        z = solve(R,x.T)
        log_sqrt_det_rho = np.sum(np.log(np.diag(R)))
        y = np.exp(-0.5 * np.sum( np.power(z.T,2) - np.power(x,2) , axis=1 ) - log_sqrt_det_rho)
        return y

    def cdf_gumbel(u,v,theta):
        """Compute CDF of Gumbel copula"""
        if theta == 1:
            pass
        else:
            h = np.power(-np.log(u),theta)+np.power(-np.log(v),theta)
            h = -np.power(h,1.0/theta)
            return np.exp(h)



    def du_copula(u,v,theta,cname):
        """Compute partial derivative of each copula function
        :param theta: single parameter of the Archimedean copula
        :param cnameL name of the copula function
        """
        if cname == '1':
            return du_clayton(u,v,theta)
        elif cname =='2':
            return du_frank(u,v,theta)
        elif cname == '3':
            return du_gumbel(u,v,theta)
        else:
            pass

    def du_frank(u,v,theta):
        if theta == 0:
            return v
        else:
            g = lambda theta,z:-1+np.exp(-np.dot(theta,z))
            num = np.dot(g(u,theta),g(v,theta))+g(v,theta)
            den = np.dot(g(u,theta)*g(v,theta))+g(1,theta)
            return num/den

    def du_clayton(u,v,theta):
        if theta == 0:
            return v
        else:
            A = pow(u,theta)
            B = pow(u,-theta)-1
            h = 1+np.dot(A,B)
            h = pow(h,(-1-theta)/theta)
            return h

    def du_gumbel(u,v,theta):
        if theta == 1:
            return v
        else:
            p1 = cdf_gumbel(u,v,theta)
            p2 = np.power(np.power(-np.log(u),theta)+np.power(-np.log(v),theta),-1+1.0/theta)
            p3 = np.power(-np.log(u),theta-1)
            return np.dot(np.dot(p1,p2),p3)/u

    def select_copula(u,v):
        """Select best copula function based on likelihood"""
        return 1,u+v



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
        self.copula = copula
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
        tree_1 = Tree(True,self.copula, self.y_ind)
        print('finish building tree 1')
        self.vine_model.append(Tree(True))
        self.u_matrix = tree_1.new_U
        for k in range(1,self.depth):
            '''get constraints from previous tree'''
            ctr = _get_constraints(self.vine_model[k-1])
            tau = _get_tau(self.vine_model[k-1],ctr)
            self.tau_mat = tau
            tree_k = Tree(False,vine_model[k-1],self.copula, self.y_ind)
            vine_model.append(tree_k)
            self.u_matrix = tree_k.new_U



    def _get_constraints(self,tree):
        """


        """
        tree_data = tree.new_T
        constraints = np.empty[1,tree.n_nodes]
        for k in range(tree.n_nodes):
            const_k = []
            for i in range(tree.n_nodes) and k!=i:

                if tree_data[k,1]==tree_data[i,1] or tree_data[k,2]==tree_data[i,2] or tree_data[k,1]==tree_data[i,2] or tree_data[k,2]==tree_data[i,1]:
                    const_k.appned(i)
            constraints[1,k] = const_k
        return constraints

    def _get_tau(self,tree):
        tree_data = tree.new_T
        tau = np.empty([len(ctr),len(ctr)])
        for i in range(len(ctr)):
            links = ctr[i]
            for j in range(len(links)):
                ed1,ed2,ing = tree.identifyEdsIng(tree_data[i,1:2],tree_data[links[j],1:2])
                tau[i,links[j]] = scipy.stats.kendalltau(self.u_matrix[ed1,ing],self.u_matrix[ed2,ing])
        return tau



class Tree(Vine):
    """instantiate a single tree in the vine model
    :param k: level of tree
    :param prev_T: tree model of previous level
    :param new_T: current tree model
    :param new_U: conditional cdfs for next level tree
    """

    def __init__(self, k, copula, y_ind, prev_T=None):
        super(Tree,self).__init__(copula, y_ind)
        self.prev_T = prev_T
        self.level = k
        if prev_T:
            self.n_nodes = len(prev_T)-1
        else:
            self.n_nodes = self.n_var
        '''For each node, tree stores position k, node index at k, node index at k+1,tau at k,tau_mat at k, tau_mat at k+1'''
        if self.level ==1 :
            self.new_T = np.empty([self.n_nodes,6])

            self._build_first_tree()
        else:
            self.new_T = np.empty([self.n_nodes,9])

            self._build_kth_tree()
        self.new_U = self._data4next_T(self.new_T)


    def identify_eds_ing(e1,e2):
        if e1[0]== e2[0]:
            ing,e1,e2 = e1[0],e1[1],e2[1]
        elif e1[0]== e2[1]:
            ing,e1,e2 = e1[0],e1[1],e2[0]
        elif e1[1]==e2[0]:
            ing,e1,e2 = e1[1],e1[0],e2[1]
        elif e1[1]==e2[1]:
            ing,e1,e2 = e1[1],e1[0],e2[0]




    def _build_first_tree(self):
        """build the first tree with n-1 variable"""
        if self.ctype == 'cvine':
            tau_mat = self.tau_mat
            tau_mat = tau_mat - np.fill_diagonal(tau_mat,10)  #ignore variance on diagonal
            tau_y = tau_mat[:,self.y_ind]
            N = len(tau_y)
            return
        if self.ctype == 'dvine':
            '''find the pair of maximum tau'''
            tau_mat = self.tau_mat
            np.fill_diagonal(tau_mat,np.NaN)
            # print(tau_mat)

            tau_y = tau_mat[:,self.y_ind]
            temp=np.empty([self.n_var,3])
            temp[:,0] = np.arange(self.n_var)
            temp[:,1] = tau_y
            temp[:,2] = abs(tau_y)
            temp[np.isnan(temp)] = -10
            tau_sorted = temp[temp[:,2].argsort()[::-1]]
            left_ind = tau_sorted[0,0]
            right_ind = tau_sorted[1,0]
            T1 = np.array([left_ind,self.y_ind,right_ind]).astype(int)
            tau_T1 = tau_sorted[:2,1] 
            '''replace tau matrix of the selected variables as a negative number (can't be selected again)'''
            tau_mat[:,[T1]] = -10
            # print(tau_mat)
            '''greedily build the rest of the first tree'''
            for k in xrange(3,self.n_nodes):
                print(k)
                valL,left=np.max(tau_mat[T1[0],:]),np.argmax(tau_mat[T1[0],:])
                valR,right=np.max(tau_mat[T1[-1],:]),np.argmax(tau_mat[T1[-1],:])
                if valL>valR:
                    '''add nodes to the left'''
                    T1=np.append(left,T1)
                    tau_T1=np.append(valL,tau_T1)
                    tau_mat[:,left]= -10
                else:
                    '''add node to the right'''
                    T1=np.append(T1,right)
                    tau_T1=np.append(tau_T1,valR)
                    tau_mat[:,right]= -10
            for k in xrange(self.n_nodes-1):
                self.new_T[k,0]=k
                self.new_T[k,1],self.new_T[k,2]=T1[k],T1[k+1]
                self.new_T[k,3]=tau_T1[k]
                '''Select copula function based on upper and lower tail functions'''
                # self.new_T[k,4],self.new_T[k,5]=Copula.select_copula(self.u_matrix[:,T1[k]],self.u_matrix[:,T1[k+1]])



    def _build_kth_tree(self):
        """build tree for level k"""
        if self.ctype == 'cvine':
            return 
        if self.ctype == 'dvine':
            for k in xrange(len(self.prev_T)-1):
                self.new_T[k,0]=k
                self.new_T[k,1],self.new_T[k,2]=k,k+1
                self.new_T[k,3]=tau_mat[k,k+1]
                [ed1,ed2,ing] = identify_eds_ing(prev_T[k,1:2],prev_T[k+1,1:2])
                U1 = self.dat_U[ed1,ing]
                U2 = self.dat_U[ed2,ing]
                self.new_T[k,4],self.new_T[k,5] = Copula.select_copula(U1,U2,self.new_T[k,3])
                self.new_T[k,6],self.new_T[k,7] = ed1, ed2
                self.new_T[k,8] = ing

    
    def _data4next_T(self,tree):
        U = np.empty([tree.shape[0],tree.shape[0]])
        for k in xrange(tree.shape[0]):
            copula_name = tree[k,4]
            copula_para = tree[k,5]
            if self.level == 1:
                U1,U2 = self.u_matrix[:,tree[k,1]],self.u_matrix[:,tree[k,2]]
            else:
                U1,U2 = self.u_matrix[tree[k,6],tree[k,8]],self.u_matrix[tree[k,7],tree[k,8]]
            '''compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/dui'''
            U1givenU2 = Copula.du_copula(U2,U1,copula_para,copula_name)
            U2givenU1 = Copula.du_copula(U1,U2,copula_para,copula_name)
            '''correction of 0 or 1'''
            U[tree[k,1],tree[k,2]]=U1givenU2
            U[tree[k,2],tree[k,1]]=U2givenU1
        return U




if __name__ == '__main__':
    copula = Copula('lucas0_train.csv','kde','dvine')
    dvine = Vine(copula,11)
    dvine.train_vine()
    print(dvine.vine_model)
                











