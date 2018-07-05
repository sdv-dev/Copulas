class RVine(VineCopula):
    """ Class for a vine copula model """

    def __init__(self):
        super(RVine, self).__init__()

    def fit(self,data):
        """Fit vine model to the data
		Returns:
		self.param: param of copula family, tree_data if model is a vine
		"""
        self.
		self.model = vine
		self.param = vine.vine_model

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

class Tree():
	"""instantiate a single tree in the vine model
	:param k: level of tree
	:param prev_T: tree model of previous level
	:param tree_data: current tree model
	:param new_U: conditional cdfs for next level tree
	"""

	def __init__(self,k,n,tau_mat,prev_T):
		# super(Tree,self).__init__(copula, y_ind)
		self.level = k+1
		self.prev_T = prev_T
		self.n_nodes = n
		self.edge_set = []
		self.tau_mat = tau_mat
		if self.level == 1 :
			self.u_matrix = prev_T
			self._build_first_tree()
			# self.print_tree()
		else:
			# self.u_matrix = prev_T.u_matrix
			self._build_kth_tree()
		self._data4next_T()

	def identify_eds_ing(self,e1,e2):
		"""find nodes connecting adjacent edges
		:param e1: pair of nodes representing edge1
		:param e2: pair of nodes representing edge2
		:output ing: nodes connecting e1 and e2
		:output n1,n2: the other node of e1 and e2 respectively
		"""
		A = set([e1.L,e1.R])
		A.update(e1.D)
		B = set([e2.L,e2.R])
		B.update(e2.D)
		D = list(A&B)
		left = list(A^B)[0]
		right = list(A^B)[1]
		return left,right,D

	def check_adjacency(self,e1,e2):
		"""check if two edges are adjacent"""
		return (e1.L==e2.L or e1.L==e2.R or e1.R==e2.L or e1.R==e2.R)

	def check_contraint(self,e1,e2):
		full_node = set([e1.L,e1.R,e2.L,e2.R])
		full_node.update(e1.D)
		full_node.update(e2.D)
		return (len(full_node)==(self.level+1))

	def _get_constraints(self):
		"""get neighboring edges
		"""
		for k in range(len(self.edge_set)):
			for i in range(len(self.edge_set)):
				#add to constriants if i shared an edge with k
				if k!=i and self.check_adjacency(self.edge_set[k],self.edge_set[i]):
					self.edge_set[k].neighbors.append(i)


	def _get_tau(self):
		"""Get tau matrix for adjacent pairs
		:param tree: a tree instance
		:param ctr: map of edge->adjacent edges
		"""
		tau = np.empty([len(self.edge_set),len(self.edge_set)])
		for i in range(len(self.edge_set)):
			for j in self.edge_set[i].neighbors:
				# ed1,ed2,ing = tree.identify_eds_ing(self.edge_set[i],self.edge_set[j])
				edge = self.edge_set[i].parent
				l_p = edge[0]
				r_p = edge[1]
				if self.level == 1:
					U1,U2 = self.u_matrix[:,l_p],self.u_matrix[:,r_p]
				else:
					U1,U2 = self.prev_T.edge_set[l_p].U,self.prev_T.edge_set[r_p].U
				tau[i,j],pvalue = scipy.stats.kendalltau(U1,U2)
		return tau


	def _build_first_tree(self):
		"""build the first tree with n-1 variable
		"""
		tau_mat = self.tau_mat
        #Prim's algorithm
		neg_tau = -1.0*abs(tau_mat)
		X=set()
		X.add(0)
		itr=0
		while len(X)!=self.n_nodes:
			adj_set=set()
			for x in X:
				for k in range(self.n_nodes):
					if k not in X and k!=x:
						adj_set.add((x,k))
			#find edge with maximum
			edge = sorted(adj_set, key=lambda e:neg_tau[e[0]][e[1]])[0]
			cop = copula.Copula(self.u_matrix[:,edge[0]],self.u_matrix[:,edge[1]])
			name,param=cop.select_copula(cop.U,cop.V)
			new_edge = Edge(itr,edge[0],edge[1],tau_mat[edge[0],edge[1]],name,param)
			new_edge.parent.append(edge[0])
			new_edge.parent.append(edge[1])
			self.edge_set.append(new_edge)
			X.add(edge[1])
			itr+=1


	def _build_kth_tree(self):
		"""build tree for level k
		"""
		neg_tau = -abs(self.tau_mat)
		visited=set()
		unvisited = set(range(self.n_nodes))
		visited.add(0) #index from previous edge set
		unvisited.remove(0)
		itr=0
		while len(visited)!=self.n_nodes:
			adj_set=set()
			for x in visited:
				for k in range(self.n_nodes):
					if k not in visited and k!=x:
						#check if (x,k) is a valid edge in the vine
						if self.check_contraint(self.prev_T.edge_set[x],self.prev_T.edge_set[k]):
							adj_set.add((x,k))
			#find edge with maximum tau
			# print('processing edge:{0}'.format(x))
			if len(list(adj_set)) == 0:
				visited.add(list(unvisited)[0])
				continue
			edge = sorted(adj_set, key=lambda e:neg_tau[e[0]][e[1]])[0]

			[ed1,ed2,ing]=self.identify_eds_ing(self.prev_T.edge_set[edge[0]],self.prev_T.edge_set[edge[1]])
			# U1 = self.u_matrix[ed1,ing]
			# U2 = self.u_matrix[ed2,ing]
			l_p = edge[0]
			r_p = edge[1]
			U1,U2 = self.prev_T.edge_set[l_p].U,self.prev_T.edge_set[r_p].U
			cop = copula.Copula(U1,U2,self.tau_mat[edge[0],edge[1]])
			name,param=cop.select_copula(cop.U,cop.V)
			new_edge = Edge(itr,ed1,ed2,self.tau_mat[edge[0],edge[1]],name,param)
			new_edge.D = ing
			new_edge.parent.append(edge[0])
			new_edge.parent.append(edge[1])
			# new_edge.likelihood = np.log(cop.pdf(U1,U2,param))
			self.edge_set.append(new_edge)
			visited.add(edge[1])
			unvisited.remove(edge[1])
			itr+=1



	def _data4next_T(self):
		"""
		prepare conditional U matrix for next tree
		"""
		# U = np.empty([self.n_nodes,self.n_nodes],dtype=object)
		edge_set = self.edge_set
		for k in range(len(edge_set)):
			edge = edge_set[k]
			copula_name = c_map[edge.name]
			copula_para = edge.param
			if self.level == 1:
				U1,U2 = self.u_matrix[:,edge.L],self.u_matrix[:,edge.R]
			else:
				prev_T = self.prev_T.edge_set
				l_p = edge.parent[0]
				r_p = edge.parent[1]
				U1,U2 = prev_T[l_p].U,prev_T[r_p].U
			'''compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/dui'''
			U1=[x for x in U1 if x is not None]
			U2=[x for x in U2 if x is not None]

			c1= copula.Copula(U2,U1,theta=copula_para,cname=copula_name,dev=True)
			U1givenU2 = c1.derivative(U2,U1,copula_para)
			U2givenU1 = c1.derivative(U1,U2,copula_para)

			'''correction of 0 or 1'''
			U1givenU2[U1givenU2==0],U2givenU1[U2givenU1==0]=eps,eps
			U1givenU2[U1givenU2==1],U2givenU1[U2givenU1==1]=1-eps,1-eps
			edge.U = U1givenU2


	def _likehood_T(self,U):
		"""Compute likelihood of the tree given an U matrix
		"""
		# newU = np.
		newU = np.empty([self.vine.n_var,self.vine.n_var])
		tree = self.tree_data
		values = np.zeros([1,tree.shape[0]])
		for i in range(tree.shape[0]):
			cname = self.vine.c_map[int(tree[i,4])]
			v1 = int(tree[i,1])
			v2 = int(tree[i,2])
			copula_para = tree[i,5]
			if self.level == 1:
				U_arr = np.array([U[v1]])
				V_arr = np.array([U[v2]])
				cop = copula.Copula(U_arr,V_arr,theta=copula_para,cname=cname,dev=True)
				values[0,i]=cop.pdf(U_arr,V_arr,copula_para)
				U1givenU2 = cop.derivative(V_arr,U_arr,copula_para)
				U2givenU1 = cop.derivative(U_arr,V_arr,copula_para)
			else:
				v1 = int(tree[i,6])
				v2 = int(tree[i,7])
				joint = int(tree[i,8])
				U1 = np.array([U[v1,joint]])
				U2 = np.array([U[v2,joint]])
				cop = copula.Copula(U1,U2,theta=copula_para,cname=cname,dev=True)
				values[0,i] = cop.pdf(U1,U2,theta=copula_para)
				U1givenU2 = cop.derivative(U2,U1,copula_para)
				U2givenU1 = cop.derivative(U1,U2,copula_para)
			newU[v1,v2]=U1givenU2
			newU[v2,v1]=U2givenU1
		# print(values)
		value = np.sum(np.log(values))
		return newU,value

	def print_tree(self):
		for e in self.edge_set:
			print(e.L,e.R,e.D,e.parent)



class Edge(object):
	def __init__(self,index,left,right,tau,copula_name,copula_para):
		self.index = index #index of the edge in the current tree
		self.level = None  #in which level of tree
		self.L = left  #left_node index
		self.R = right #right_node index
		self.D = [] #dependence_set
		self.parent = [] #indices of parent edges in the previous tree
		self.tau = tau   #correlation of the edge
		self.name = copula_name
		self.param = copula_para
		self.U = None
		self.likelihood = None
		self.neighbors = []

	def get_likehood(U):
		"""Compute likelihood given a U matrix
		"""
		if self.level==1:
			cop = copula.Copula(U[:,self.L],self.u_matrix[:,self.U])
			name,param=cop.select_copula(cop.U,cop.V)
			self.likelihood = cop.pdf(U_arr,V_arr,copula_para)
		else:
			[ed1,ed2,ing]=self.identify_eds_ing(self.prev_T.edge_set[edge[0]],self.prev_T.edge_set[edge[1]])
			# U1 = self.u_matrix[ed1,ing]
			# U2 = self.u_matrix[ed2,ing]
			l_p = edge[0]
			r_p = edge[1]
			U1,U2 = self.prev_T.edge_set[l_p].U,self.prev_T.edge_set[r_p].U
			cop = copula.Copula(U1,U2,self.tau_mat[edge[0],edge[1]])
			name,param=cop.select_copula(cop.U,cop.V)
