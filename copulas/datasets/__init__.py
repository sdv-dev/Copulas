import numpy as np
import pandas as pd
from scipy import stats

from copulas import random_seed


def load_three_dimensional(seed=42):
    """
    This dataset contains 6 columns, each of which corresponds to a different
    univariate distribution:

        bernoulli - a Bernoulli distribution with p=0.3
        bimodal - a mixture of two Gaussians at 0.0 and 10.0 with stdev=1
        uniform - a uniform distribution in [-1.0, 3.0]
        normal - a normal distribution at 1.0 with stdev=1
        constant - a constant value
        exponential - an exponential distribution at 3.0 with rate 1.0
    """
    data = np.zeros((1000, 3))
    with random_seed(seed):
        data[:, 0] = stats.beta.rvs(a=0.1, b=0.1, size=data.shape[0])
        data[:, 1] = stats.beta.rvs(a=0.1, b=0.5, size=data.shape[0])
        data[:, 2] = np.random.normal(size=data.shape[0]) + data[:, 1] * 10
    return pd.DataFrame(data, columns=["x", "y", "z"])


def load_diverse_univariates(seed=42):
    """
    This dataset contains 6 columns, each of which corresponds to a different
    univariate distribution:

        bernoulli - a Bernoulli distribution with p=0.3
        bimodal - a mixture of two Gaussians at 0.0 and 10.0 with stdev=1
        uniform - a uniform distribution in [-1.0, 3.0]
        normal - a normal distribution at 1.0 with stdev=1
        constant - a constant value
        exponential - an exponential distribution at 3.0 with rate 1.0
    """
    size = 1000
    df = pd.DataFrame()
    with random_seed(seed):
        df["bernoulli"] = (np.random.random(size=size) < 0.3).astype(float)
        df["bimodal"] = np.random.normal(size=size) * df["bernoulli"] + \
            np.random.normal(size=size, loc=10) * (1.0 - df["bernoulli"])
        df["uniform"] = 4.0 * np.random.random(size=size) - 1.0
        df["normal"] = np.random.normal(size=size, loc=1.0)
        df["constant"] = np.random.random()  # a single random number
        df["exponential"] = np.random.exponential(size=size) + 3.0
    return df
