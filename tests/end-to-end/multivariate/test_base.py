import numpy as np
import pytest

from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import GaussianMultivariate, VineCopula


class TestMultivariate():

    @pytest.mark.parametrize('model', [GaussianMultivariate(), VineCopula('direct')])
    def test_fixed_random_state(self, model):
        """Test that the multivariate models work with a fixed seed.

        Expect that fixing the seed generates a reproducable sequence
        of samples. Expect that these samples are different from randomly
        sampled results.
        """
        data = sample_trivariate_xyz()
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
