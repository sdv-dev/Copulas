import MVCopula
import scipy.stats as st
import pandas as pd
import numpy as np
from univariate import GaussianUnivariate

class GaussianCopula(MVCopula):
	""" Class for a gaussian copula model """

	def __init__(self, data):
		super(GaussianCopula, self).__init__()
		self.cols = {}
		self.fit()

	def fit(self):
		keys = self.data.keys()
		for key in keys:
			self.cols[keys] = GaussianUnivariate(data[key].as_matrix())

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
	gc = GaussianCopula(data)
