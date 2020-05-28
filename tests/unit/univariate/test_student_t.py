from unittest import TestCase

import numpy as np
from scipy.stats import t

from copulas.univariate.student_t import StudentTUnivariate


class TestStudentTUnivariate(TestCase):

    def test__fit_constant(self):
        distribution = StudentTUnivariate()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'df': 100,
            'loc': 1,
            'scale': 0
        }

    def test__fit(self):
        distribution = StudentTUnivariate()

        data = t.rvs(size=1000, df=3, loc=1, scale=1)
        distribution._fit(data)

        assert distribution._params == {
            'df': 2.8331725136645227,
            'loc': 1.0151500951225847,
            'scale': 1.000966233180422,
        }

    def test__is_constant_true(self):
        distribution = StudentTUnivariate()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = StudentTUnivariate()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()
