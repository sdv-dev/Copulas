import scipy.stats as st
import scipy.integrate as integrate
import pandas as pd
import numpy as np
from copulas.multivariate.MVCopula import MVCopula
from copulas.univariate.GaussianUnivariate import GaussianUnivariate

class GaussianCopula(MVCopula):
	""" Class for a gaussian copula model """

	def __init__(self):
		super(GaussianCopula, self).__init__()
		self.distribs = {}
		self.cov_matrix = None
		self.data = None
		self.means = None
		self.pdf = None
		self.cdf = None
		self.ppf = None

	def fit(self, data, distrib_map = None):
		self.data = data
		keys = data.keys()
		# create distributions based on user input
		if distrib_map is not None:
			for key in distrib_map:
				# this isn''t fully working yet
				self.distribs[key] = distrib_map[key](data[key].as_matrix())
		else:
			for key in keys:
				self.distribs[key] = GaussianUnivariate()
				self.distribs[key].fit(data[key].as_matrix())
		self.cov_matrix, self.means, self.distribution = self._get_covariance_matrix()
		self.pdf = st.multivariate_normal.pdf

	def _get_covariance_matrix(self):
		res = self.data.copy()
		# loops through columns and applies transformation
		for col in self.data.keys():
			X = self.data.loc[:,col]
			distrib = self.distribs[col]
			# get original distrib's cdf of the column
			cdf = distrib.get_cdf(X)
			print(cdf)
			# get inverse cdf using standard normal
			res.loc[:,col] = st.norm.ppf(cdf)
		print(res.as_matrix())
		means = [np.mean(res.iloc[:,i].as_matrix()) for i in range(res.shape[1])]
		return (np.cov(res.as_matrix()), means, res)

	def get_pdf(self, X):
		cov = self.cov_matrix * np.identity(3)
		return self.pdf(X, self.means, cov)

	def get_cdf(self,X):
		func = lambda *args: self.get_pdf([args[i] for i in range(len(args))])
		ranges = [[-10000,val] for val in X]
		return integrate.nquad(func, ranges)[0]

	def sample(self, num_rows=1):
		res = {}
		# means = [self.distribs[self.data.iloc[:,i].name].mean for i in range(self.data.shape[1])]
		means = [0.0]*len(self.cov_matrix)
		samples = np.random.multivariate_normal(means, self.cov_matrix, size=(num_rows,))
		print(samples)
		# run through cdf and inverse cdf
		for i in range(self.data.shape[1]):
			label = self.data.iloc[:,i].name
			distrib = self.distribs[label]
			# use standard normal's cdf
			res[label] = st.norm.cdf(samples[:,i])
			# use original distributions inverse cdf
			res[label] = distrib.inverse_cdf(res[label])
		return pd.DataFrame(data=res)
