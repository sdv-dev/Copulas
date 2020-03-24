import os
import tempfile
from unittest import TestCase

from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import VineCopula


class TestGaussian(TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_fit_sample_direct(self):
        data = sample_trivariate_xyz()
        model = VineCopula('direct')
        model.fit(data)

        for N in [10, 50, 100]:
            assert len(model.sample(N)) == N

        sampled_data = model.sample(10)

        assert sampled_data.shape == (10, 3)
        for column in data.columns:
            assert column in sampled_data

    def test_fit_sample_regular(self):
        data = sample_trivariate_xyz()
        model = VineCopula('regular')
        model.fit(data)

        sampled_data = model.sample(10)
        assert sampled_data.shape == (10, 3)

    def test_fit_sample_center(self):
        data = sample_trivariate_xyz()
        model = VineCopula('center')
        model.fit(data)

        sampled_data = model.sample(10)
        assert sampled_data.shape == (10, 3)

    def test_to_dict_from_dict(self):
        data = sample_trivariate_xyz()
        model = VineCopula('direct')
        model.fit(data)

        sampled_data = model.sample(10)

        params = model.to_dict()
        model2 = VineCopula.from_dict(params)

        sampled_data = model2.sample(10)
        assert sampled_data.shape == (10, 3)

    def test_save_load(self):
        data = sample_trivariate_xyz()
        model = VineCopula('direct')
        model.fit(data)

        sampled_data = model.sample(10)

        path_to_model = os.path.join(self.test_dir.name, "model.pkl")
        model.save(path_to_model)
        model2 = VineCopula.load(path_to_model)

        sampled_data = model2.sample(10)
        assert sampled_data.shape == (10, 3)
