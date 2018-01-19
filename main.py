import argparse


def main(d):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Copula Model')
    parser.add_argument('data', help ='name of the casual dataset')
    parser.add_argument('-utype', action ='store',default ='kde',help='utype')
    parser.add_argument('-ctype', action ='store',default = 'gaussian',help='ctype')
    # parser.add_argument('-cname', action ='store',default = 'clayton',help ='cname')

    args = parser.parse_args()

    d = Copula(args.data,args.utype,args.ctype)
    

