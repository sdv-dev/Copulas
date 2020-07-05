from copulas.datasets import sample_trivariate_xyz
from copulas.visualization import compare_3d


def test_compare_3d():
    data = sample_trivariate_xyz()

    compare_3d(data, data)
