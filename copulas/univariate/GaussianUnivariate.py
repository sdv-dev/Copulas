import numpy as np
from scipy.stats import norm
from copulas.univariate.UnivariateDistrib import UnivariateDistrib

class GaussianUnivariate(UnivariateDistrib):
	""" Gaussian univariate model """

	def __init__(self):
		super(GaussianUnivariate, self).__init__()
		self.column = None
		self.mean = 0
		self.std = 1
		self.min = -np.inf
		self.max = np.inf

	def fit(self, column):
		self.column = column
		self.mean = np.mean(column)
		self.std = np.std(column)
		self.max = max(column)
		self.min = min(column)

	def get_pdf(self, x):
		return norm.pdf(x, loc=self.mean, scale=self.std)

	def get_cdf(self, x):
		return norm.cdf(x, loc=self.mean, scale=self.std)

	# def _calculate_cdf(self):
	# 	def cdf(data):
	# 		u = []
	# 		for y in data:
	# 			ui = self.pdf.integrate_box_1d(-np.inf, y)
	# 			u.append(ui)
	# 		u = np.asarray(u)
	# 		return u
	# 	return cdf

	def inverse_cdf(self, u):
		""" given a cdf value, returns a value in original space """
		return norm.ppf(u, loc=self.mean, scale=self.std)

	def sample(self):
		""" returns new data point based on model """
		raise NotImplementedError