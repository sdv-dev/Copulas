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
    # LOGGER.debug(gc.get_pdf(np.array([1,5,9])))
    # LOGGER.debug(gc.get_cdf([2, 5, 8]))
