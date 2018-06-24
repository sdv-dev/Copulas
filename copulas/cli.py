# -*- coding: utf-8 -*-

"""Console script for copulas."""
import sys

import click

from copulas.multivariate.models import CopulaModel


@click.command()
@click.argument('data')
@click.argument('utype', default='kde')
@click.argument('ctype', default='dvine')
def main(data, utype, ctype):
    """Create a Vine from the data, utype and ctype"""
    copula = CopulaModel(data, utype, ctype)
    click.echo(copula.sampling(1, plot=True))
    click.echo(copula.model.vine_model[-1].tree_data)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
