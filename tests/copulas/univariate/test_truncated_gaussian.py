from unittest import TestCase

from scipy.stats import truncnorm
from scipy.stats.distributions import rv_frozen

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

        # Generate data from a truncated gaussian
        loc, scale = 2.0, 1.0
        min_value, max_value = -1.0, 4.0
        a = (min_value - loc) / scale
        b = (max_value - loc) / scale
        data = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=1000000, random_state=42)

        # Run
        instance.fit(data)

        # Check
        assert isinstance(instance.model, rv_frozen)
        self.assertAlmostEqual(min_value, instance.min, places=2)
        self.assertAlmostEqual(max_value, instance.max, places=2)
        self.assertAlmostEqual(loc, instance.mean, places=2)
        self.assertAlmostEqual(scale, instance.std, places=2)

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
