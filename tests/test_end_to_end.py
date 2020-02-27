import os
import tempfile
from unittest import TestCase

import numpy as np

from copulas.datasets import load_diverse_univariates, load_three_dimensional
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianKDE


class TestEndToEnd(TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_basic(self):
        for dataloader in [load_diverse_univariates, load_three_dimensional]:
            with self.subTest(dataloader=dataloader, univariate="default"):
                self._kitchen_sink(dataloader())

    def _kitchen_sink(self, dataset):
        """
        For the given dataset, this runs "everything but the kitchen sink" (i.e.
        every feature of GaussianMultivariate that is officially supported) and
        makes sure it doesn't crash.
        """
        model = GaussianMultivariate({
            dataset.columns[0]: GaussianKDE()  # Use a KDE for the first column
        })
        model.fit(dataset)
        for N in [10, 1000, 100]:
            assert len(model.sample(N)) == N
        sampled_data = model.sample(10)
        pdf = model.probability_density(sampled_data)
        cdf = model.cumulative_distribution(sampled_data)

        # Test Save/Load from Dictionary
        config = model.to_dict()
        model2 = GaussianMultivariate.from_dict(config)

        print(model.to_dict())
        print(model2.to_dict())

        for N in [10, 1000, 100]:
            assert len(model2.sample(N)) == N
        pdf2 = model2.probability_density(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))

        path_to_model = os.path.join(self.test_dir.name, "model.pkl")
        model.save(path_to_model)
        model2 = GaussianMultivariate.load(path_to_model)
        for N in [10, 1000, 100]:
            assert len(model2.sample(N)) == N
        pdf2 = model2.probability_density(sampled_data)
        cdf2 = model2.cumulative_distribution(sampled_data)
        assert np.all(np.isclose(pdf, pdf2, atol=0.01))
        assert np.all(np.isclose(cdf, cdf2, atol=0.01))
