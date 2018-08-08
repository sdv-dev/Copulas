from unittest import TestCase

from copulas.bivariate import Gumbel
from copulas.bivariate.base import Bivariate, CopulaTypes


class TestGumbel(TestCase):

    def test_sample(self):
        """After being fit, copula can produce samples."""
        # Setup
        copula = Bivariate(CopulaTypes.GUMBEL)

        U = [0.1, 0.2, 0.3, 0.4]
        V = [0.5, 0.6, 0.5, 0.8]

        copula.fit(U, V)

        # Run
        result = copula.sample(10)

        # Check
        assert copula.__class__ == Gumbel
        assert result.shape == (10, 2)
