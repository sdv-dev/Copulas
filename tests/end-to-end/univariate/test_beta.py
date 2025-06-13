import os
import tempfile
from unittest import TestCase

import numpy as np
import pytest
from scipy.stats import beta

from copulas.univariate import BetaUnivariate


class TestGaussian(TestCase):
    def setUp(self):
        self.data = beta.rvs(a=1.0, b=1.0, loc=1.0, scale=1.0, size=50000)
        self.constant = np.full(100, fill_value=5)
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_fit_sample(self):
        model = BetaUnivariate()
        model.fit(self.data)

        np.testing.assert_allclose(model._params['loc'], 1.0, atol=0.2)
        np.testing.assert_allclose(model._params['scale'], 1.0, atol=0.2)
        np.testing.assert_allclose(model._params['a'], 1.0, atol=0.2)
        np.testing.assert_allclose(model._params['b'], 1.0, atol=0.2)

        sampled_data = model.sample(50)

        assert isinstance(sampled_data, np.ndarray)
        assert sampled_data.shape == (50,)

    def test_fit_sample_constant(self):
        model = BetaUnivariate()
        model.fit(self.constant)

        sampled_data = model.sample(50)

        assert isinstance(sampled_data, np.ndarray)
        assert sampled_data.shape == (50,)

        assert model._constant_value == 5
        np.testing.assert_equal(np.full(50, 5), model.sample(50))

    def test_fit_raises(self):
        """Test it for dataset that fails."""
        model = BetaUnivariate()
        data = np.array([  # From GH #472
            3.337169,
            6.461266,
            4.871053,
            4.206772,
            5.157541,
            3.437069,
            6.712143,
            5.135402,
            6.453203,
            4.623059,
            5.827161,
            5.291858,
            5.571134,
            5.441359,
            4.816826,
            3.277817,
            4.215228,
            4.48338,
            4.345968,
            6.125759,
            4.860464,
            6.511877,
            3.959057,
            4.882996,
            6.058503,
            3.337436,
            5.06921,
            4.414371,
            4.564768,
            5.1014,
            4.161663,
            5.757317,
            4.032375,
            3.907653,
            4.269559,
            4.08505,
            6.811531,
            5.02597,
            5.438358,
            3.44442,
            3.462209,
            4.871354,
            5.947369,
            4.167546,
            4.692054,
            5.542011,
            4.926634,
            4.491286,
            5.344663,
            4.526089,
            1.645776,
        ])

        err_msg = 'Converged parameters for beta distribution have a near-zero range.'
        with pytest.raises(ValueError, match=err_msg):
            model.fit(data)

    def test_pdf(self):
        model = BetaUnivariate()
        model.fit(self.data)

        sampled_data = model.sample(50)

        # Test PDF
        pdf = model.probability_density(sampled_data)
        assert (0 < pdf).all()

    def test_cdf(self):
        model = BetaUnivariate()
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
        model = BetaUnivariate()
        model.fit(self.data)

        sampled_data = model.sample(50)

        params = model.to_dict()
        model2 = BetaUnivariate.from_dict(params)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))

    def test_to_dict_from_dict_constant(self):
        model = BetaUnivariate()
        model.fit(self.constant)

        sampled_data = model.sample(50)
        pdf = model.probability_density(sampled_data)
        cdf = model.cumulative_distribution(sampled_data)

        params = model.to_dict()
        model2 = BetaUnivariate.from_dict(params)

        np.testing.assert_equal(np.full(50, 5), sampled_data)
        np.testing.assert_equal(np.full(50, 5), model2.sample(50))
        np.testing.assert_equal(np.full(50, 1), pdf)
        np.testing.assert_equal(np.full(50, 1), model2.probability_density(sampled_data))
        np.testing.assert_equal(np.full(50, 1), cdf)
        np.testing.assert_equal(np.full(50, 1), model2.cumulative_distribution(sampled_data))

    def test_to_dict_constant(self):
        model = BetaUnivariate()
        model.fit(self.constant)

        params = model.to_dict()

        assert params == {
            'type': 'copulas.univariate.beta.BetaUnivariate',
            'loc': 5,
            'scale': 0,
            'a': 1,
            'b': 1,
        }

    def test_save_load(self):
        model = BetaUnivariate()
        model.fit(self.data)

        sampled_data = model.sample(50)

        path_to_model = os.path.join(self.test_dir.name, 'model.pkl')
        model.save(path_to_model)
        model2 = BetaUnivariate.load(path_to_model)

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
        model = BetaUnivariate()
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
