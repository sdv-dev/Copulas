import os

import pandas as pd

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def make_dataset():
    """Handcrafted dataset to test bivariate cdf."""
    rows = []
    for x0 in [0.33, 0.47, 0.61]:
        for x1 in [0.2, 0.33, 0.71, 0.9]:
            rows.append({"x0": x0, "x1": x1})

    return pd.DataFrame(rows)


if __name__ == '__main__':
    dataset = make_dataset()

    name = os.path.basename(__file__).replace(".py", ".csv")
    path = os.path.join(ROOT, 'input', name)

    dataset.to_csv(path, index=False)
