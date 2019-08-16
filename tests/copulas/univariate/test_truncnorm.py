from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from copulas import EPSILON
from copulas.univariate.truncnorm import TruncNorm


class TestTruncNorm(TestCase):

    def test_init(self):
        """On init, there are no errors and attributes are set as expected."""
        # Run / Setup
        instance = TruncNorm()

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

    @patch('copulas.univariate.base.scipy.stats.truncnorm', autospec=True)
    def test_fit(self, truncnorm_mock):
        """On fit, the attribute model is set with an instance of scipy.stats.truncnorm."""
        # Setup
        instance = TruncNorm()
        data = np.array([1, 2, 3, 4, 5])

        model_mock = MagicMock(
            ppf='percent_point',
            rvs='sample',
            pdf='probability_density',
            cdf='cumulative_distribution'
        )
        truncnorm_mock.return_value = model_mock

        # Run
        instance.fit(data)

        # Check
        assert instance.model == model_mock
        assert instance.probability_density == 'probability_density'
        assert instance.percent_point == 'percent_point'
        assert instance.cumulative_distribution == 'cumulative_distribution'
        assert instance.sample == 'sample'

        truncnorm_mock.assert_called_once_with(
            1 - EPSILON, 5 + EPSILON, loc=3.0, scale=1.4142135623730951)

    def test_from_dict_unfitted(self):
        """from_dict creates a new instance from a dict of params."""
        # Setup
        parameters = {
            'type': 'copulas.univariate.truncnorm.TruncNorm',
            'fitted': False,
        }

        # Run
        instance = TruncNorm.from_dict(parameters)

        # Check
        assert instance.fitted is False
        assert instance.constant_value is None
        assert instance.model is None

    def test_from_dict_fitted(self):
        """from_dict creates a new instance from a dict of params."""
        # Setup
        parameters = {
            'type': 'copulas.univariate.truncnorm.TruncNorm',
            'fitted': True,
            'a': -10,
            'b': 10,
            'mean': 0,
            'std': 1
        }

        # Run
        instance = TruncNorm.from_dict(parameters)

        # Check
        assert instance.fitted is True
        assert instance.constant_value is None
        assert instance.model.a == -10
        assert instance.model.b == 10
        assert instance.model.kwds['loc'] == 0
        assert instance.model.kwds['scale'] == 1

    def test__fit_params(self):
        """_fit_params returns a dict with the params of the scipy model."""
        # Setup
        data = np.array(range(5))
        instance = TruncNorm()
        instance.fit(data)

        expected_result = {
            'a': - EPSILON,
            'b': 4 + EPSILON,
            'loc': 2.0,
            'scale': 1.4142135623730951
        }

        # Run
        result = instance._fit_params()

        # Check
        assert result == expected_result
