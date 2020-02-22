import os

import pandas as pd

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def make_dataset():
    """Handcrafted dataste to test bivariate fit."""
    # FIXME: This exists here only to show where to put any code
    # used to generate an input dataset. If values are written by
    # hand, it should be enough to say so in the test JSON config.
    return pd.DataFrame({
        'x0': [0.2, 0.2, 0.4, 0.6, 0.8, 0.8],
        'x1': [0.1, 0.3, 0.5, 0.4, 0.6, 0.9]
    })


if __name__ == '__main__':
    dataset = make_dataset()

    name = os.path.basename(__file__)[:-len('.py')]
    path = os.path.join(ROOT, 'input', name + '.csv')

    dataset.to_csv(path, index=False)
