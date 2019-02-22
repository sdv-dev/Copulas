from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import scipy

from copulas.univariate.truncnorm import TruncNorm


class TestTruncNorm(TestCase):

    def test_init(self):
        """On init, there are no errors and attributes are set as expected."""
        # Run / Setup
        instance = TruncNorm()

        # Check
        assert instance.model_class == 'truncnorm'
        assert instance.model_fit_init is True
        assert instance.method_map == {
            'probability_density': 'pdf',
            'cumulative_distribution': 'cdf',
            'percent_point': 'ppf',
            'sample': 'rvs'
        }
        assert instance.model is None
        assert instance.fitted is False
        assert instance.constant_value is None

    @patch('copulas.univariate.base.scipy.stats', autospec=True)
    def test_fit(self, scipy_mock):
        """On fit, the attribute model is set with an instance of scipy.stats.truncnorm."""
        # Setup
        instance = TruncNorm()
        data = np.array([1, 2, 3, 4, 5])

        truncnorm = MagicMock(spec=scipy.stats.truncnorm)
        truncnorm.return_value = 'a truncnorm model'
        scipy_mock.truncnorm = truncnorm

        # Run
        instance.fit(data)

        # Check
        assert instance.model == 'a truncnorm model'

        assert callable(instance.probability_density)
        assert callable(instance.cumulative_distribution)
        assert callable(instance.percent_point)
        assert callable(instance.sample)

        scipy_mock.assert_not_called()
        truncnorm.assert_called_once_with(1, 5)

    def test_from_dict_unfitted(self):
        """from_dict creates a new instance from a dict of params."""
        # Setup
        parameters = {
            'type': 'copulas.univariate.truncnorm.TruncNorm',
            'fitted': False,
            'constant_value': None
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
            'constant_value': None,
            'a': 0,
            'b': 10
        }

        # Run
        instance = TruncNorm.from_dict(parameters)

        # Check
        assert instance.fitted is True
        assert instance.constant_value is None
        assert instance.model.a == 0
        assert instance.model.b == 10

    def test__fit_params(self):
        """_fit_params returns a dict with the params of the scipy model."""
        # Setup
        data = np.array(range(5))
        instance = TruncNorm()
        instance.fit(data)

        expected_result = {
            'a': 0,
            'b': 4
        }

        # Run
        result = instance._fit_params()

        # Check
        assert result == expected_result
