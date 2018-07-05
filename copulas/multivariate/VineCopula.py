import numpy as np
from copulas.multivariate.MVCopula import MVCopula

Class VineCopula(MVCopula):
    """ Class for a vine copula model """

    def __init__(self):
        super(VineCopula, self).__init__()
        self.u_matrix = None

        self.cdf = None
        self.ppf = None

		self.model = None
		self.param = None

    def fit(self,data):
        """Fit copula model to the data
		Returns:
		self.param: param of copula family, tree_data if model is a vine
		"""
        self.data = data
        self.tau_mat = self.data.corr(method='kendall').as_matrix()
        self.u_matrix = u_mat

        self.truncated = truncated
        self.n_sample = self.data.shape[0]
		self.n_var = self.data.shape[1]
        self.depth = self.n_var - 1

		self.vine_model = []
		model = RVine(u_mat,tau_mat,ppf,truncated)
        model.train_vine()

    def sampling(self,n,out_dir=None,plot=False):
		sampled = np.zeros([n,self.n_var])
		for i in range(n):
			x = self.model._sampling(n)
			sampled[i,:]=x
		if plot:
			plt.scatter(self.model_data.ix[:, 0],self.model_data.ix[:, 1],c='green')
			plt.scatter(sampled[:,0],sampled[:,1],c='red')
			plt.show()
		if out_dir:
			np.savetxt(out_dir, sampled, delimiter=",")
		return sampled
