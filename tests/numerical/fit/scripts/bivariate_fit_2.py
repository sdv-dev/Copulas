import os

import pandas as pd

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def make_dataset():
    """Handcrafted dataste to test bivariate fit."""
    # FIXME: This exists here only to show where to put any code
    # used to generate an input dataset. If values are written by
    # hand, it should be enough to say so in the test JSON config.

    return pd.DataFrame({
        'x0': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.56],
        'x1': [0.57, 0.5, 0.53, 0.55, 0.51, 0.54, 0.56]
    })


if __name__ == '__main__':
    dataset = make_dataset()

    name = os.path.basename(__file__)[:-len('.py')]
    path = os.path.join(ROOT, 'input', name + '.csv')

    dataset.to_csv(path, index=False)
