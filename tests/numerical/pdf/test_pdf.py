import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

from copulas import get_instance

BASE = os.path.dirname(__file__)
TESTS = glob.glob(BASE + '/test_cases/*/*.json')


@pytest.mark.parametrize("config_path", TESTS)
def test_pdf(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Setup
    test_obj = config['test']
    instance = get_instance(test_obj['class'], **test_obj['kwargs'])
    data = pd.read_csv(os.path.join(BASE, 'input', config['input']['points']))
    output_R = pd.read_csv(os.path.join(BASE, 'output', config['output']['R']))
    output_M = pd.read_csv(os.path.join(BASE, 'output', config['output']['M']))

    # Run
    instance.theta = config['input']['theta']

    # Asserts
    pdfs = instance.pdf(data.values)

    rtol = config['settings']['rtol']

    assert np.all(np.isclose(output_R["pdf"], pdfs, rtol=rtol)), config_file
    assert np.all(np.isclose(output_M["pdf"], pdfs, rtol=rtol)), config_file
