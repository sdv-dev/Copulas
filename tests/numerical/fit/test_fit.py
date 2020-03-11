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
def test_fit(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Setup
    test_obj = config['test']
    instance = get_instance(test_obj['class'], **test_obj['kwargs'])
    data = pd.read_csv(os.path.join(BASE, 'input', config['test_case_inputs']['points']))

    # Run
    instance.fit(data.values)

    # Asserts
    params = instance.to_dict()

    rtol = config['settings']['rtol']

    for other, expected in config['expected_output'].items():
        for key, exp in expected.items():
            obs = params[key]
            msg = "Mismatch against {} on {}".format(other, config_path)
            assert np.isclose(exp, obs, rtol=rtol), msg
