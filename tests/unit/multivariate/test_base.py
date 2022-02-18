import numpy as np

from copulas.multivariate.base import Multivariate


class TestMultivariate:

    def test_set_random_state(self):
        """Test `set_random_state` works as expected"""
        # Setup
        instance = Multivariate()

        # Run
        instance.set_random_state(3)

        # Check
        assert isinstance(instance.random_state, np.random.RandomState)
