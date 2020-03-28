import os
import tempfile
from unittest import TestCase

import numpy as np

from copulas.datasets import sample_univariate_bimodal
from copulas.univariate import GaussianKDE


class TestGaussian(TestCase):

    def setUp(self):
        self.data = sample_univariate_bimodal()
        self.constant = np.full(100, fill_value=5)
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_fit_sample(self):
        model = GaussianKDE()
        model.fit(self.data)

        sampled_data = model.sample(50)

        assert isinstance(sampled_data, np.ndarray)
        assert sampled_data.shape == (50, )

    def test_fit_sample_constant(self):
        model = GaussianKDE()
        model.fit(self.constant)

        sampled_data = model.sample(50)

        assert isinstance(sampled_data, np.ndarray)
        assert sampled_data.shape == (50, )

        assert model._constant_value == 5
        np.testing.assert_equal(np.full(50, 5), model.sample(50))

    def test_pdf(self):
        model = GaussianKDE()
        model.fit(self.data)

        sampled_data = model.sample(50)

        # Test PDF
        pdf = model.probability_density(sampled_data)
        assert (0 < pdf).all()

    def test_cdf(self):
        model = GaussianKDE()
        model.fit(self.data)

        sampled_data = model.sample(50)

        # Test the CDF
        cdf = model.cumulative_distribution(sampled_data)
        assert (0 < cdf).all() and (cdf < 1).all()

        # Test CDF increasing function
        sorted_data = sorted(sampled_data)
        cdf = model.cumulative_distribution(sorted_data)
        assert (np.diff(cdf) >= 0).all()

    def test_to_dict_from_dict(self):
        model = GaussianKDE()
        model.fit(self.data)

        sampled_data = model.sample(50)

        params = model.to_dict()
        model2 = GaussianKDE.from_dict(params)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))

    def test_to_dict_sample_size(self):
        model = GaussianKDE(sample_size=10)
        model.fit(self.constant)

        params = model.to_dict()

        assert params['type'] == 'copulas.univariate.gaussian_kde.GaussianKDE'
        assert len(params['dataset']) == 10

    def test_to_dict_constant(self):
        model = GaussianKDE()
        model.fit(self.constant)

        params = model.to_dict()

        assert params == {
            'type': 'copulas.univariate.gaussian_kde.GaussianKDE',
            'dataset': [5] * 100
        }

    def test_save_load(self):
        model = GaussianKDE()
        model.fit(self.data)

        sampled_data = model.sample(50)

        path_to_model = os.path.join(self.test_dir.name, "model.pkl")
        model.save(path_to_model)
        model2 = GaussianKDE.load(path_to_model)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))
