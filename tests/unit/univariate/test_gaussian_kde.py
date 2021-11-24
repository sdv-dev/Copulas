#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `univariate.gaussian_kde` module."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.stats import gaussian_kde

from copulas.datasets import sample_univariate_bimodal
from copulas.univariate.gaussian_kde import GaussianKDE


class TestGaussianKDE(TestCase):

    def test__get_model_no_sample_size(self):
        self = MagicMock()
        self._sample_size = None
        self._params = {
            'dataset': np.array([1, 2, 3, 4, 5])
        }
        self.bw_method = None
        self.weights = None
        model = GaussianKDE._get_model(self)

        assert isinstance(model, gaussian_kde)
        assert self._sample_size == 5
        np.testing.assert_allclose(model.dataset, np.array([[1, 2, 3, 4, 5]]))

    def test__get_model_sample_size(self):
        self = MagicMock()
        self._sample_size = 3
        self._params = {
            'dataset': np.array([1, 2, 3, 4, 5])
        }
        self.bw_method = None
        self.weights = None
        model = GaussianKDE._get_model(self)

        assert isinstance(model, gaussian_kde)
        assert self._sample_size == 3
        np.testing.assert_allclose(model.dataset, np.array([[1, 2, 3, 4, 5]]))

    def test__get_bounds(self):
        self = MagicMock()
        self._params = {
            'dataset': np.array([1, 2, 3, 4, 5])
        }

        lower, upper = GaussianKDE._get_bounds(self)

        k = 5 * np.std([1, 2, 3, 4, 5])
        assert lower == 1 - k
        assert upper == 5 + k

    def test__fit_constant(self):
        distribution = GaussianKDE()

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'dataset': [1, 1, 1, 1],
        }

    def test__fit_constant_sample_size(self):
        distribution = GaussianKDE(sample_size=3)

        distribution._fit_constant(np.array([1, 1, 1, 1]))

        assert distribution._params == {
            'dataset': [1, 1, 1],
        }

    def test__fit(self):
        distribution = GaussianKDE()

        distribution._fit(np.array([1, 2, 3, 4]))

        assert distribution._params == {
            'dataset': [1, 2, 3, 4],
        }

    def test__fit_sample_size(self):
        distribution = GaussianKDE(sample_size=3)

        distribution._fit(np.array([1, 2, 3, 4]))

        assert len(distribution._params['dataset']) == 1
        assert len(distribution._params['dataset'][0]) == 3

    def test__is_constant_true(self):
        distribution = GaussianKDE()

        distribution.fit(np.array([1, 1, 1, 1]))

        assert distribution._is_constant()

    def test__is_constant_false(self):
        distribution = GaussianKDE()

        distribution.fit(np.array([1, 2, 3, 4]))

        assert not distribution._is_constant()

    def test_cumulative_distribution(self):
        """cumulative_distribution evaluates with the model."""
        instance = GaussianKDE()
        instance.fit(np.array([0.9, 1.0, 1.1]))

        cdf = instance.cumulative_distribution(np.array([
            0.0,  # There is no data below this (cdf = 0.0).
            1.0,  # Half the data is below this (cdf = 0.5).
            2.0,  # All the data is below this (cdf = 1.0).
            -1.0  # There is no data below this (cdf = 0).
        ]))

        assert np.all(np.isclose(cdf, np.array([0.0, 0.5, 1.0, 0.0]), atol=1e-3))

    def test_percent_point(self):
        """percent_point evaluates with the model."""
        instance = GaussianKDE()
        instance.fit(np.array([0.5, 1.0, 1.5]))

        cdf = instance.percent_point(np.array([0.001, 0.5, 0.999]))

        assert cdf[0] < 0.0, 'The 0.001th percentile should be small.'
        assert abs(cdf[1] - 1.0) < 0.1, 'The 50% percentile should be the median.'
        assert cdf[2] > 2.0, 'The 0.999th percentile should be large.'

    def test_percent_point_bisect(self):
        """percent_point evaluates with the model."""
        instance = GaussianKDE()
        instance.fit(np.array([0.5, 1.0, 1.5]))

        cdf = instance.percent_point(np.array([0.001, 0.5, 0.999]), method='bisect')

        assert cdf[0] < 0.0, 'The 0.001th percentile should be small.'
        assert abs(cdf[1] - 1.0) < 0.1, 'The 50% percentile should be the median.'
        assert cdf[2] > 2.0, 'The 0.999th percentile should be large.'

    def test_percent_point_invalid_value(self):
        """Evaluating an invalid value will raise ValueError."""
        fit_data = np.array([1, 2, 3, 4, 5])
        instance = GaussianKDE()
        instance.fit(fit_data)

        error_msg = r'Expected values in range \[0.0, 1.0\].'
        with pytest.raises(ValueError, match=error_msg):
            instance.percent_point(np.array([2.]))

    def test_percent_point_invertibility(self):
        instance = GaussianKDE()
        instance.fit(sample_univariate_bimodal())
        cdf = np.random.random(size=1000)
        x = instance.percent_point(cdf)
        assert np.abs(instance.cumulative_distribution(x) - cdf).max() < 1e-6

    def test_percent_point_boundary_values(self):
        instance = GaussianKDE()
        instance.fit(np.array([0.0, 0.5, 1.0]))
        x = instance.percent_point(np.array([0.0, 1.0]))
        assert x[0] == float('-inf')
        assert x[1] == float('inf')

    @patch('copulas.univariate.gaussian_kde.gaussian_kde', autospec=True)
    def test_sample(self, kde_mock):
        """Sample calls the gaussian_kde.resample method."""
        instance = GaussianKDE()
        instance.fit(np.array([1, 2, 3, 4]))

        model = kde_mock.return_value
        model.resample.return_value = np.array([[1, 2, 3]])

        samples = instance.sample(3)

        instance._model.resample.assert_called_once_with(3)
        np.testing.assert_equal(samples, np.array([1, 2, 3]))

    @patch('copulas.univariate.gaussian_kde.gaussian_kde', autospec=True)
    def test_probability_density(self, kde_mock):
        """Sample calls the gaussian_kde.resample method."""
        instance = GaussianKDE()
        instance.fit(np.array([1, 2, 3, 4]))

        model = kde_mock.return_value
        model.evaluate.return_value = np.array([0.1, 0.2, 0.3])

        pdf = instance.probability_density(np.array([1, 2, 3]))

        assert instance._model.evaluate.call_count == 1
        input_array = instance._model.evaluate.call_args[0][0]
        np.testing.assert_equal(input_array, np.array([1, 2, 3]))
        np.testing.assert_equal(pdf, np.array([0.1, 0.2, 0.3]))

    def test__extract_constant(self):
        distribution = GaussianKDE()
        distribution._params = {
            'dataset': [1, 1, 1, 1],
        }

        constant = distribution._extract_constant()

        assert 1 == constant
