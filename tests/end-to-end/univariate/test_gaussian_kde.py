import os
import tempfile
from unittest import TestCase

import numpy as np
from scipy.stats import ks_2samp, norm, randint

from copulas.datasets import sample_univariate_bimodal
from copulas.multivariate import GaussianMultivariate
from copulas.univariate.gaussian_kde import GaussianKDE


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
        assert (0 <= cdf).all()
        assert (cdf <= 1).all()

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

    def test_to_dict_from_dict_constant(self):
        model = GaussianKDE()
        model.fit(self.constant)

        sampled_data = model.sample(50)
        pdf = model.probability_density(sampled_data)
        cdf = model.cumulative_distribution(sampled_data)

        params = model.to_dict()
        model2 = GaussianKDE.from_dict(params)

        np.testing.assert_equal(np.full(50, 5), sampled_data)
        np.testing.assert_equal(np.full(50, 5), model2.sample(50))
        np.testing.assert_equal(np.full(50, 1), pdf)
        np.testing.assert_equal(np.full(50, 1), model2.probability_density(sampled_data))
        np.testing.assert_equal(np.full(50, 1), cdf)
        np.testing.assert_equal(np.full(50, 1), model2.cumulative_distribution(sampled_data))

    def test_save_load(self):
        model = GaussianKDE()
        model.fit(self.data)

        sampled_data = model.sample(50)

        path_to_model = os.path.join(self.test_dir.name, 'model.pkl')
        model.save(path_to_model)
        model2 = GaussianKDE.load(path_to_model)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))

    def test_gaussiankde_arguments(self):
        size = 1000
        low = 0
        high = 9
        data = randint.rvs(low, high, size=size) + norm.rvs(0, 0.1, size=size)
        dist = GaussianMultivariate(distribution=GaussianKDE(bw_method=0.01))
        dist.fit(data)
        samples = dist.sample(size).to_numpy()[0]
        d, p = ks_2samp(data, samples)
        assert p >= 0.05

    def test_fixed_random_state(self):
        """Test that the univariate models work with a fixed seed.
        Expect that fixing the seed generates a reproducable sequence
        of samples. Expect that these samples are different from randomly
        sampled results.
        """
        model = GaussianKDE()
        model.fit(self.data)

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
