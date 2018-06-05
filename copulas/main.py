import argparse
import logging

from copulas.models import CopulaModel, Vine

LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copula Vine Model')
    parser.add_argument('data', help='name of the dataset')
    parser.add_argument('-utype', action='store', default='kde', help='utype')
    parser.add_argument('-ctype', action='store', default='dvine', help='ctype')
    # parser.add_argument('-cname', action ='store',default = 'clayton',help ='cname')

    args = parser.parse_args()
    data = CopulaModel(args.data, args.utype, args.ctype)
    dvine = Vine(data, 11)
    dvine.train_vine()
    LOGGER.debug(dvine.vine_model[-1].tree_data)
