import pandas as pd

from copulas.multivariate.GaussianCopula import GaussianCopula

if __name__ == '__main__':
    data = pd.read_csv('data/iris.data.csv')
    gc = GaussianCopula()
    gc.fit(data)
    print(gc.sample(num_rows=1))
    print(gc.cov_matrix)
    # print(gc.get_pdf(np.array([1,5,9])))
    # print(gc.get_cdf([2, 5, 8]))
