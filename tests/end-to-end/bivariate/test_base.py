import numpy as np
import pytest

from copulas.bivariate import Bivariate


class TestBivariate():

    @pytest.mark.parametrize('model', Bivariate.subclasses())
    def test_fixed_random_state(self, model):
        """Test that the bivariate models work with a fixed seed.

        Expect that fixing the seed generates a reproducable sequence
        of samples. Expect that these samples are different from randomly
        sampled results.
        """
        data = np.array([
            [0.2, 0.1],
            [0.2, 0.2],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.4, 0.9],
            [0.6, 0.4],
            [0.6, 0.8],
            [0.7, 0.6],
            [0.7, 0.9],
            [0.8, 0.9],
        ])
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
