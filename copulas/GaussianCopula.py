from MVCopula import MVCopula
import scipy.stats as st
import scipy.integrate as integrate
import pandas as pd
import numpy as np
from univariate.GaussianUnivariate import GaussianUnivariate

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

	def fit(self, data):
		self.data = data
		keys = data.keys()
		for key in keys:
			self.distribs[key] = GaussianUnivariate(data[key].as_matrix())
		self.cov_matrix, self.means = self._get_covariance_matrix()
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
		# for row in range(self.data.shape[0]):
		# 	for col in self.data.keys():
		# 		x = self.data.loc[row,col]
		# 		distrib = self.distribs[col]
		# 		cdf = distrib.get_cdf(x)
		# 		res.loc[row, col] = distrib.inverse_cdf(cdf) # TODO: this should be self.ppf
		print(res.as_matrix())
		means = [np.mean(res.iloc[:,i].as_matrix()) for i in range(res.shape[1])]
		return (np.cov(res.as_matrix()), means)

	def get_pdf(self, X):
		cov = self.cov_matrix * np.identity(3)
		return self.pdf(X, self.means, cov)

	def get_cdf(self,X):
		func = lambda a,b,c: self.get_pdf([X[0],X[1],X[2]])
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

	# def _calculate_cdf(self):
	# 	def cdf(data):
	# 		u = []
	# 		np_data = data.as_matrix()
	# 		lower_bounds = [-np.inf]*len(data[0])
	# 		for i in range(len(np_data)):
	# 			upper_bounds = np_data[i]
	# 			ui = self.pdf.integrate_box(lower_bounds, higher_bounds)
	# 			u.append(ui)
	# 		u = np.asarray(u)
	# 		return u
	# 	return cdf

if __name__ == '__main__':
	data = pd.read_csv('example.csv')
	gc = GaussianCopula()
	gc.fit(data)
	print(gc.sample(num_rows = 1))
	print(gc.cov_matrix)
	print(gc.get_pdf(np.array([1,5,9])))
	print(gc.get_cdf([2, 5, 8]))
