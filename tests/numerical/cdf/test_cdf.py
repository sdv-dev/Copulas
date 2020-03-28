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

    inputs = config['test_case_inputs']
    outputs = config['expected_output']
    input_points = pd.read_csv(os.path.join(BASE, 'input', inputs['points']))
    output_r = pd.read_csv(os.path.join(BASE, 'output', outputs['R']))
    output_matlab = pd.read_csv(os.path.join(BASE, 'output', outputs['Matlab']))

    # Run
    instance.theta = inputs['theta']

    # Asserts
    cdfs = instance.cdf(input_points.values)

    rtol = config['settings']['rtol']

    assert np.all(np.isclose(output_r["cdf"], cdfs, rtol=rtol))
    assert np.all(np.isclose(output_matlab["cdf"], cdfs, rtol=rtol))
