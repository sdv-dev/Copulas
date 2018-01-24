import scipy

from copulalib import Copula
import numpy as np
import pandas as pd

import utils



class CopulaException(Exception):
    pass

class CopulaUtil(object):
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

    @staticmethod
    def cdf_gumbel(u,v,theta):
        """Compute CDF of Gumbel copula"""
        if theta == 1:
            pass
        else:
            h = np.power(-np.log(u),theta)+np.power(-np.log(v),theta)
            h = -np.power(h,1.0/theta)
            return np.exp(h)


    def du_copula(self,u,v,theta,cname):
        """Compute partial derivative of each copula function
        :param theta: single parameter of the Archimedean copula
        :param cnameL name of the copula function
        """
        if cname == 1:
            return self.du_clayton(u,v,theta)
        elif cname ==2:
            return self.du_frank(u,v,theta)
        elif cname == 3:
            return self.du_gumbel(u,v,theta)
        else:
            pass

    def du_frank(self,u,v,theta):
        if theta == 0:
            return v
        else:
            g = lambda theta,z:-1+np.exp(-np.dot(theta,z))
            num = np.multiply(g(u,theta),g(v,theta))+g(v,theta)
            den = np.multiply(g(u,theta),g(v,theta))+g(1,theta)
            return num/den

    def du_clayton(self,u,v,theta):
        if theta == 0:
            return v
        else:
            A = pow(u,theta)
            B = pow(u,-theta)-1
            h = 1+np.multiply(A,B)
            h = pow(h,(-1-theta)/theta)
            return h

    def du_gumbel(self,u,v,theta):
        if theta == 1:
            return v
        else:
            p1 = cdf_gumbel(u,v,theta)
            p2 = np.power(np.power(-np.log(u),theta)+np.power(-np.log(v),theta),-1+1.0/theta)
            p3 = np.power(-np.log(u),theta-1)
            return np.dot(np.dot(p1,p2),p3)/u

    def select_copula(self,u,v,tau=None):
        """Select best copula function based on likelihood"""

        theta = [0]*3
        if not tau:
            tau = scipy.stats.kendalltau(u,v)
        if tau < 0:
            bestC = 2
            copula = Copula(u,v,family ='frank')
            paramC = copula.theta
        else:
            theta[0] = Copula(u,v,family='clayton').theta
            theta[1] = Copula(u,v,family='frank').theta
            theta[2] = Copula(u,v,family='gumbel').theta
            bestC = 2
            paramC = theta[1]
        return bestC,paramC




                











