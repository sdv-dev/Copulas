# -*- coding: utf-8 -*-

"""Console script for copulas."""
import argparse

from copulas.multivariate.VineCopula import CopulaModel


def main(data, utype, ctype):
    """Create a Vine from the data, utype and ctype"""
    copula = CopulaModel(data, utype, ctype)
    print(copula.sampling(1, plot=True))
    print(copula.model.vine_model[-1].tree_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copulas Example')
    parser.add_argument('--utype', default='kde')
    parser.add_argument('--ctype', default='dvine')
    parser.add_argument('data')

    args = parser.parse_args()

    main(args.data, args.utype, args.ctype)
