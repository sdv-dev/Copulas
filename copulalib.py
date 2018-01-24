#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 19:13:28 2011

@ author:                  Sat Kumar Tomer 
@ author's webpage:        http://civil.iisc.ernet.in/~satkumar/
@ author's email id:       satkumartomer@gmail.com
@ author's website:        www.ambhas.com

"""
from __future__ import division
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fmin
import sys
import statistics as st
from scipy.interpolate import interp1d

#from scipy.stats import *
#import random as R
#from pylab import*
#from scipy.interpolate import interp1d


class Copula():
    """
    This class estimate parameter of copula
    generate joint random variable for the parameters
    This class has following three copulas:
        Clayton
        Frank
        Gumbel
        
    Example:
        x = np.random.normal(size=100)
        y = np.random.normal(size=100)
        foo = Copula(x, y, 'frank')
        u,v = foo.generate(100)
    """
    

    def __init__(self, X, Y, family):
        """ initialise the class with X and Y
        Input:
            X:        one dimensional numpy array
            Y:        one dimensional numpy array
            family:   clayton or frank or gumbel
            
            Note: the size of X and Y should be same
        """
        # check dimension of input arrays
        eps = sys.float_info.epsilon

        if not ((X.ndim==1) and (Y.ndim==1)):
            raise ValueError('The dimension of array should be one.')
        
        # input array should have same zie
        if X.size != Y.size:
            raise ValueError('The size of both array should be same.')
        
        # check if the name of copula family correct
        copula_family = ['clayton', 'frank', 'gumbel']
        if family not in copula_family:
            raise ValueError('The family should be clayton or frank or gumbel')
        
        self.X = X
        self.Y = Y
        self.family = family
        
        # estimate Kendall'rank correlation
        tau = kendalltau(self.X, self.Y)[0]
        self.tau = tau   
        # estimate pearson R and spearman R
        self.pr = pearsonr(self.X, self.Y)[0]
        self.sr = spearmanr(self.X, self.Y)[0]
        
        # estimate the parameter of copula
        self._get_parameter()
        
        # set U and V to none
        self.U = None
        self.V = None
        
        
    def _get_parameter(self):
        """ estimate the parameter (theta) of copula
        """        
        
        if self.family == 'clayton':
            self.theta = 2*self.tau/(1-self.tau)
            
        elif self.family == 'frank':
            self.theta = -fmin(self._frank_fun, -5, disp=False)[0]
            
        elif self.family == 'gumbel':
            self.theta = 1/(1-self.tau)
    
    def generate_uv(self, n=1000):
        """
        Generate random variables (u,v)
        
        Input:
            n:        number of random copula to be generated
        
        Output:
            U and V:  generated copula
            
        """
        # CLAYTON copula
        if self.family == 'clayton':
            U = np.random.uniform(size = n)
            W = np.random.uniform(size = n)
                
            if self.theta <= -1:
                raise ValueError('the parameter for clayton copula should be more than -1')
            elif self.theta==0:
                raise ValueError('The parameter for clayton copula should not be 0')
                
            if self.theta < sys.float_info.epsilon :
                V = W
            else:
                V = U*(W**(-self.theta/(1 + self.theta)) - 1 + U**self.theta)**(-1/self.theta)
    
        # FRANK copula
        elif self.family == 'frank':
            U = np.random.uniform(size = n)
            W = np.random.uniform(size = n)
            
            if self.theta == 0:
                raise ValueError('The parameter for frank copula should not be 0')
            
            if abs(self.theta) > np.log(sys.float_info.max):
                V = (U < 0) + np.sign(self.theta)*U
            elif abs(self.theta) > np.sqrt(sys.float_info.epsilon):
                V = -np.log((np.exp(-self.theta*U)*(1-W)/W + np.exp(-self.theta)
                            )/(1 + np.exp(-self.theta*U)*(1-W)/W))/self.theta
            else:
                V = W
        
        # GUMBEL copula
        elif self.family == 'gumbel':
            if self.theta <= 1 :
                raise ValueError('the parameter for GUMBEL copula should be greater than 1')
            if self.theta < 1 + sys.float_info.epsilon:
                U = np.random.uniform(size = n)
                V = np.random.uniform(size = n)
            else:
                u = np.random.uniform(size = n)
                w = np.random.uniform(size = n)
                w1 = np.random.uniform(size = n)
                w2 = np.random.uniform(size = n)
                
                u = (u - 0.5) * np.pi
                u2 = u + np.pi/2;
                e = -np.log(w)
                t = np.cos(u - u2/self.theta)/ e
                gamma = (np.sin(u2/self.theta)/t)**(1/self.theta)*t/np.cos(u)
                s1 = (-np.log(w1))**(1/self.theta)/gamma
                s2 = (-np.log(w2))**(1/self.theta)/gamma
                U = np.array(np.exp(-s1))
                V = np.array(np.exp(-s2))
        
        self.U = U
        self.V = V        
        return U,V
    
    def generate_xy(self, n=1000):
        """
        Generate random variables (x, y)
        
        Input:
            n:        number of random copula to be generated
        
        Output:
            X1 and Y1:  generated copula random numbers
            
        """
        # if U and V are not already generated
        if self.U is None:
            self.generate_uv(n)
            
        # estimate inverse cdf of x andy
        self._inverse_cdf()
        
        # estimate X1 and Y1        
        X1 = self._inv_cdf_x(self.U)
        Y1 = self._inv_cdf_y(self.V)
        
        return X1, Y1
        
        
    def _inverse_cdf(self):
        """
        This module will calculate the inverse of CDF 
        which will be used in getting the ensemble of X and Y from
        the ensemble of U and V
        
        The statistics module is used to estimate the CDF, which uses
        kernel methold of cdf estimation
        
        To estimate the inverse of CDF, interpolation method is used, first cdf 
        is estimated at 100 points, now interpolation function is generated 
        to relate cdf at 100 points to data
        """
        x2, x1 = st.cpdf(self.X, kernel = 'Epanechnikov', n = 100)
        self._inv_cdf_x = interp1d(x2, x1)
        
        y2, y1 = st.cpdf(self.Y, kernel = 'Epanechnikov', n = 100)
        self._inv_cdf_y = interp1d(y2, y1)
        
    
    def _integrand_debye(self,t):
         """ 
         Integrand for the first order debye function
         """
         return t/(np.exp(t)-1)
         
    def _debye(self, alpha):
        """
        First order Debye function
        """
        return quad(self._integrand_debye, sys.float_info.epsilon, alpha)[0]/alpha
    
    def _frank_fun(self, alpha):
        """
        optimization of this function will give the parameter for the frank copula
        """
        diff = (1-self.tau)/4.0  - (self._debye(-alpha)-1)/alpha
        return diff**2
        
        








