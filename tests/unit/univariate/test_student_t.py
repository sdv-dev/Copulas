from unittest import TestCase

import numpy as np
from scipy.stats import t

from copulas.univariate.student_t import StudentTUnivariate


class TestStudentTUnivariate(TestCase):

    def test__fit(self):
        distribution = StudentTUnivariate()

        data = t.rvs(size=50000, df=3, loc=1, scale=1)
        distribution._fit(data)

        expected = {
            'df': 3,
            'loc': 1,
            'scale': 1,
        }
        for key, value in distribution._params.items():
            np.testing.assert_allclose(value, expected[key], rtol=0.3)

    def test__is_constant_true(self):
        distribution = StudentTUnivariate()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = StudentTUnivariate()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()
