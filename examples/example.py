import logging

import pandas as pd

from copulas.multivariate.GaussianCopula import GaussianCopula

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    data = pd.read_csv('data/iris.data.csv')
    gc = GaussianCopula()
    gc.fit(data)
    LOGGER.debug(gc.sample(num_rows=1))
    LOGGER.debug(gc.cov_matrix)
