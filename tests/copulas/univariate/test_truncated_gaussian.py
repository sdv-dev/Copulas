from unittest import TestCase

import numpy as np
from scipy.stats.distributions import rv_frozen

from copulas import EPSILON
from copulas.univariate.truncated_gaussian import TruncatedGaussian


class TestTruncatedGaussian(TestCase):

    def test_init(self):
        """On init, there are no errors and attributes are set as expected."""
        # Run / Setup
        instance = TruncatedGaussian()

        # Check
        assert instance.model_class == 'truncnorm'
        assert instance.unfittable_model is True
        assert instance.probability_density == 'pdf'
        assert instance.cumulative_distribution == 'cdf'
        assert instance.percent_point == 'ppf'
        assert instance.sample == 'rvs'
        assert instance.model is None
        assert instance.fitted is False
        assert instance.constant_value is None

    def test_fit(self):
        """On fit, the attribute model is set with an instance of scipy.stats.truncnorm."""
        # Setup
        instance = TruncatedGaussian()
        data = np.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4])

        # Run
        instance.fit(data)

        # Check
        assert isinstance(instance.model, rv_frozen)
        assert instance.min == 0 - EPSILON
        assert instance.max == 4 + EPSILON
        assert instance.mean == 2
        assert instance.std == 1

    def test_from_dict_unfitted(self):
        """from_dict creates a new instance from a dict of params."""
        # Setup
        parameters = {
            'type': 'copulas.univariate.truncnorm.TruncatedGaussian',
            'fitted': False,
        }

        # Run
        instance = TruncatedGaussian.from_dict(parameters)

        # Check
        assert instance.fitted is False
        assert instance.constant_value is None
        assert instance.model is None

    def test_from_dict_fitted(self):
        """from_dict creates a new instance from a dict of params."""
        # Setup
        parameters = {
            'type': 'copulas.univariate.truncnorm.TruncatedGaussian',
            'fitted': True,
            'min': -10,
            'max': 10,
            'std': 1,
            'mean': 1,
        }

        # Run
        instance = TruncatedGaussian.from_dict(parameters)

        # Check
        assert instance.fitted is True
        assert isinstance(instance.model, rv_frozen)
        assert instance.constant_value is None
        assert instance.min == -10
        assert instance.max == 10
        assert instance.std == 1
        assert instance.mean == 1

    def test_from_dict_constant(self):
        """from_dict creates a new instance from a dict of params."""
        # Setup
        parameters = {
            'type': 'copulas.univariate.truncnorm.TruncatedGaussian',
            'fitted': True,
            'min': 10,
            'max': 10,
            'std': 0,
            'mean': 10,
        }

        # Run
        instance = TruncatedGaussian.from_dict(parameters)

        # Check
        assert instance.fitted is True
        assert instance.constant_value == 10

    def test__fit_params(self):
        """_fit_params returns a dict with the params of the scipy model."""
        # Setup
        instance = TruncatedGaussian()
        instance.min = 0
        instance.max = 4
        instance.std = 1
        instance.mean = 2

        expected_result = {
            'min': 0,
            'max': 4,
            'std': 1,
            'mean': 2,
        }

        # Run
        result = instance._fit_params()

        # Check
        assert result == expected_result
