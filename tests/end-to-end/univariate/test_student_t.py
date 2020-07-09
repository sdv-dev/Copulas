import os
import tempfile
from unittest import TestCase

import numpy as np
from scipy.stats import t

from copulas.univariate import StudentTUnivariate


class TestStudentT(TestCase):

    def setUp(self):
        self.data = t.rvs(df=3.0, loc=1.0, scale=0.5, size=50000)
        self.constant = np.full(100, fill_value=5)
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_fit_sample(self):
        model = StudentTUnivariate()
        model.fit(self.data)

        np.testing.assert_allclose(model._params['df'], 3.0, rtol=0.2)
        np.testing.assert_allclose(model._params['loc'], 1.0, atol=0.2)
        np.testing.assert_allclose(model._params['scale'], 0.5, atol=0.2)

        sampled_data = model.sample(50)

        assert isinstance(sampled_data, np.ndarray)
        assert sampled_data.shape == (50, )

    def test_fit_sample_constant(self):
        model = StudentTUnivariate()
        model.fit(self.constant)

        sampled_data = model.sample(50)

        assert isinstance(sampled_data, np.ndarray)
        assert sampled_data.shape == (50, )

        assert model._constant_value == 5
        np.testing.assert_allclose(np.full(50, 5), model.sample(50), atol=0.2)

    def test_pdf(self):
        model = StudentTUnivariate()
        model.fit(self.data)

        sampled_data = model.sample(50)

        # Test PDF
        pdf = model.probability_density(sampled_data)
        assert (0 < pdf).all()

    def test_cdf(self):
        model = StudentTUnivariate()
        model.fit(self.data)

        sampled_data = model.sample(50)

        # Test CDF
        cdf = model.cumulative_distribution(sampled_data)
        assert (0 < cdf).all() and (cdf < 1).all()

        # Test CDF increasing function
        sorted_data = sorted(sampled_data)
        cdf = model.cumulative_distribution(sorted_data)
        assert (np.diff(cdf) >= 0).all()

    def test_to_dict_from_dict(self):
        model = StudentTUnivariate()
        model.fit(self.data)

        sampled_data = model.sample(50)

        params = model.to_dict()
        model2 = StudentTUnivariate.from_dict(params)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))

    def test_to_dict_constant(self):
        model = StudentTUnivariate()
        model.fit(self.constant)

        params = model.to_dict()

        df = params.pop('df')
        assert np.isclose(df, 152801, atol=1)

        assert params == {
            'type': 'copulas.univariate.student_t.StudentTUnivariate',
            'loc': 5,
            'scale': 0,
        }

    def test_save_load(self):
        model = StudentTUnivariate()
        model.fit(self.data)

        sampled_data = model.sample(50)

        path_to_model = os.path.join(self.test_dir.name, "model.pkl")
        model.save(path_to_model)
        model2 = StudentTUnivariate.load(path_to_model)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))
