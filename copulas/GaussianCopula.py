from MVCopula import MVCopula
import scipy.stats as st
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

	def fit(self, data):
		self.data = data
		keys = data.keys()
		for key in keys:
			self.distribs[key] = GaussianUnivariate(data[key].as_matrix())
		self.cov_matrix = self._get_covariance_matrix()

	def _get_covariance_matrix(self):
		res = self.data.copy()
		for row in range(self.data.shape[0]):
			for col in self.data.keys():
				x = self.data.loc[row,col]
				distrib = self.distribs[col]
				cdf = distrib.get_cdf(x)
				res.loc[row, col] = distrib.inverse_cdf(cdf)
		return np.cov(res.as_matrix())

	def sample(self, num_rows=1):
		res = {}
		means = [self.distribs[self.data.iloc[:,i].name].mean for i in range(self.data.shape[1])]
		samples = np.random.multivariate_normal(means, self.cov_matrix, size=(num_rows,))
		print(means)
		print(samples)
		# run through cdf and inverse cdf
		for i in range(self.data.shape[1]):
			label = self.data.iloc[:,i].name
			distrib = self.distribs[label]
			res[label] = distrib.get_cdf(samples[:,i])
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
	print(gc.sample(num_rows = 2))
	print(gc.cov_matrix)
