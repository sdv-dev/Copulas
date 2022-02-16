import numpy as np
import pytest
from scipy.stats import norm

from copulas.univariate import Univariate


class TestUnivariate():

    @pytest.mark.parametrize('model', Univariate._get_subclasses())
    def test_fixed_random_state(self, model):
        """Test that the univariate models work with a fixed seed.

        Expect that fixing the seed generates a reproducable sequence
        of samples. Expect that these samples are different from randomly
        sampled results.
        """
        data = norm.rvs(loc=1.0, scale=0.5, size=100)
        model = model()
        model.fit(data)

        sampled_random = model.sample(10)
        model.set_random_state(0)
        sampled_0_0 = model.sample(10)
        sampled_0_1 = model.sample(10)

        model.set_random_state(0)
        sampled_1_0 = model.sample(10)
        sampled_1_1 = model.sample(10)

        assert not np.array_equal(sampled_random, sampled_0_0)
        assert not np.array_equal(sampled_0_0, sampled_0_1)
        np.testing.assert_array_equal(sampled_0_0, sampled_1_0)
        np.testing.assert_array_equal(sampled_0_1, sampled_1_1)
