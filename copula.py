import copulalib
import pandas as pd

import utils


class CopulaException(Exception):
    pass

class Copula(object):
    def __init__(self, full_path, utype, ctype, cname):
        """Instantiates an instance of the copula object

        :param utype: the distribution for the univariate, can be 'kde','norm'
        :param ctype: the type of copula models, can be 'Gaussian','cvine','dvine'
        :param cname: the choice of copulas, can be 'clayton','gumbel'  

        """
        self.utype = utype
        self.ctype = ctype
        self.cname = cname

        self.model_data = pd.read_csv(full_path, sep=',', index_col=False, 
            converters=converters, na_values=['NaN', 'nan', 'NULL', 'null'],
            low_memory=False)

    def _train_cdf(self, utype):
        """Find the cdf for each columns."""
        cdfs = []
        uni = []
        for col in self.model_data:
            dist = utils.Distribution(column=model_data[col].values)
            dist.name=utype
            cdfs.append(dist.cdf)
        return cdfs,uni


class CopulaTree(object):
    def __init__(self, ctype, uni):
        self.ctype = ctype
        self.matU = uni
        self.n_nodes = len(uni)
        self.depth = self.n_nodes - 1
        self.tauMat = self.tauMat

    def _build_first_tree(self):
        if ctype == 'cvine':
            


    def _build_kth_tree(self,k):
        '''build tree for level k
        :param k: level k
        '''
        if ctype == 'cvine':
            











