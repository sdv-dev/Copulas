import os
import tempfile
from unittest import TestCase

import numpy as np
from scipy.stats import truncnorm

from copulas.univariate import TruncatedGaussian


class TestGaussian(TestCase):

    def setUp(self):
        self.data = truncnorm.rvs(a=0.0, b=4.0, loc=1.0, scale=1.0, size=50000)
        self.constant = np.full(100, fill_value=5)
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_fit_sample(self):
        model = TruncatedGaussian(minimum=1, maximum=5)
        model.fit(self.data)

        np.testing.assert_allclose(model._params['loc'], 1.0, atol=0.2)
        np.testing.assert_allclose(model._params['scale'], 1.0, atol=0.2)
        np.testing.assert_allclose(model._params['a'], 0.0, atol=0.2)
        np.testing.assert_allclose(model._params['b'], 4.0, atol=0.2)

        sampled_data = model.sample(50)

        assert isinstance(sampled_data, np.ndarray)
        assert sampled_data.shape == (50, )

    def test_fit_sample_constant(self):
        model = TruncatedGaussian()
        model.fit(self.constant)

        sampled_data = model.sample(50)

        assert isinstance(sampled_data, np.ndarray)
        assert sampled_data.shape == (50, )

        assert model._constant_value == 5
        np.testing.assert_equal(np.full(50, 5), model.sample(50))

    def test_pdf(self):
        model = TruncatedGaussian()
        model.fit(self.data)

        sampled_data = model.sample(50)

        # Test PDF
        pdf = model.probability_density(sampled_data)
        assert (0 < pdf).all()

    def test_cdf(self):
        model = TruncatedGaussian()
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
        model = TruncatedGaussian()
        model.fit(self.data)

        sampled_data = model.sample(50)

        params = model.to_dict()
        model2 = TruncatedGaussian.from_dict(params)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))

    def test_to_dict_from_dict_constant(self):
        model = TruncatedGaussian()
        model.fit(self.constant)

        sampled_data = model.sample(50)
        pdf = model.probability_density(sampled_data)
        cdf = model.cumulative_distribution(sampled_data)

        params = model.to_dict()
        model2 = TruncatedGaussian.from_dict(params)

        np.testing.assert_equal(np.full(50, 5), sampled_data)
        np.testing.assert_equal(np.full(50, 5), model2.sample(50))
        np.testing.assert_equal(np.full(50, 1), pdf)
        np.testing.assert_equal(np.full(50, 1), model2.probability_density(sampled_data))
        np.testing.assert_equal(np.full(50, 1), cdf)
        np.testing.assert_equal(np.full(50, 1), model2.cumulative_distribution(sampled_data))

    def test_to_dict_constant(self):
        model = TruncatedGaussian(minimum=1, maximum=5)
        model.fit(self.constant)

        params = model.to_dict()

        assert params == {
            'type': 'copulas.univariate.truncated_gaussian.TruncatedGaussian',
            'loc': 5,
            'scale': 0,
            'a': 5,
            'b': 5,
        }

    def test_save_load(self):
        model = TruncatedGaussian()
        model.fit(self.data)

        sampled_data = model.sample(50)

        path_to_model = os.path.join(self.test_dir.name, 'model.pkl')
        model.save(path_to_model)
        model2 = TruncatedGaussian.load(path_to_model)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))

    def test_fixed_random_state(self):
        """Test that the univariate models work with a fixed seed.
        Expect that fixing the seed generates a reproducable sequence
        of samples. Expect that these samples are different from randomly
        sampled results.
        """
        model = TruncatedGaussian()
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
