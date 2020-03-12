import os
import tempfile
from unittest import TestCase

import numpy as np

from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import BetaUnivariate, GaussianKDE, ParametricType, Univariate


class TestGaussian(TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_fit_sample(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate()
        model.fit(data)

        for N in [10, 50, 100]:
            assert len(model.sample(N)) == N

        sampled_data = model.sample(10)

        assert sampled_data.shape == (10, 3)
        for column in data.columns:
            assert column in sampled_data

    def test_fit_sample_distribution_class(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate(GaussianKDE)
        model.fit(data)

        sampled_data = model.sample(10)
        assert sampled_data.shape == (10, 3)

    def test_fit_sample_distribution_name(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate('copulas.univariate.gaussian_kde.GaussianKDE')
        model.fit(data)

        sampled_data = model.sample(10)
        assert sampled_data.shape == (10, 3)

    def test_fit_sample_distribution_instance(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate(distribution=GaussianKDE())
        model.fit(data)

        sampled_data = model.sample(10)
        assert sampled_data.shape == (10, 3)

    def test_fit_sample_distribution_dict(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate(distribution={
            'x': GaussianKDE()
        })
        model.fit(data)

        sampled_data = model.sample(10)
        assert sampled_data.shape == (10, 3)

    def test_fit_sample_distribution_dict_multiple(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate(distribution={
            'x': Univariate(parametric=ParametricType.PARAMETRIC),
            'y': BetaUnivariate(),
            'z': GaussianKDE()
        })
        model.fit(data)

        sampled_data = model.sample(10)
        assert sampled_data.shape == (10, 3)

    def test_pdf(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate()
        model.fit(data)

        sampled_data = model.sample(10)

        # Test PDF
        pdf = model.probability_density(sampled_data)
        assert (0 < pdf).all()

    def test_cdf(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate()
        model.fit(data)

        sampled_data = model.sample(10)

        # Test CDF
        cdf = model.cumulative_distribution(sampled_data)
        assert (0 < cdf).all() and (cdf < 1).all()

        # Test CDF increasing function
        for column in sampled_data.columns:
            sorted_data = sampled_data.sort_values(column)
            other_columns = data.columns.to_list()
            other_columns.remove(column)
            row = sorted_data.sample(1).iloc[0]
            for column in other_columns:
                sorted_data[column] = row[column]

            cdf = model.cumulative_distribution(sorted_data)
            assert (np.diff(cdf) >= 0).all()

    def test_to_dict_from_dict(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate()
        model.fit(data)

        sampled_data = model.sample(10)

        params = model.to_dict()
        model2 = GaussianMultivariate.from_dict(params)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))

    def test_save_load(self):
        data = sample_trivariate_xyz()
        model = GaussianMultivariate()
        model.fit(data)

        sampled_data = model.sample(10)

        path_to_model = os.path.join(self.test_dir.name, "model.pkl")
        model.save(path_to_model)
        model2 = GaussianMultivariate.load(path_to_model)

        pdf = model.probability_density(sampled_data)
        pdf2 = model2.probability_density(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))

        cdf = model.cumulative_distribution(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))
