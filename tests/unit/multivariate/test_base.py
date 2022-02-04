from copulas.multivariate.base import Multivariate


class TestMultivariate:

    def test_set_random_seed(self):
        """Test `set_random_seed` works as expected"""
        # Setup
        instance = Multivariate()

        # Run
        instance.set_random_seed(3)

        # Check
        assert instance.random_seed == 3
