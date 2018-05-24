import argparse

import copula
import vine


def main(d):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Copula Vine Model')
    parser.add_argument('data', help ='name of the dataset')
    parser.add_argument('-utype', action ='store',default ='kde',help='utype')
    parser.add_argument('-ctype', action ='store',default = 'dvine',help='ctype')
    # parser.add_argument('-cname', action ='store',default = 'clayton',help ='cname')

    args = parser.parse_args()
    data = copula.CopulaUtil(args.data,args.utype,args.ctype)
    dvine = Vine(data,11)
    dvine.train_vine()
    print(dvine.vine_model[-1].tree_data)
    

