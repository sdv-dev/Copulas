from copulas.datasets import (
    sample_bivariate_age_income,
    sample_trivariate_xyz,
    sample_univariate_beta,
)
from copulas.visualization import compare_1d, compare_2d, compare_3d


def test_compare_1d():
    data = sample_univariate_beta()

    compare_1d(data, data)


def test_compare_2d():
    df = sample_bivariate_age_income()

    compare_2d(df, df)


def test_compare_3d():
    data = sample_trivariate_xyz()

    compare_3d(data, data)
