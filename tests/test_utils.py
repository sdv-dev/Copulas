import logging
from unittest import TestCase

import numpy as np

from copulas import utils

LOGGER = logging.getLogger(__name__)


class TestGenerateSamples(TestCase):

    def test_simple_case(self):
        # Setup
        d0 = utils.Distribution(column=np.linspace(-15, 15))
        d1 = utils.Distribution(
            column=['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
            categorical=True
        )
        d2 = utils.Distribution(
            column=['T', 'T', 'T', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
            categorical=True
        )

        covariance = np.array([[1, 0.2, 0.3], [0.2, 1, 0.5], [0.3, 0.5, 1]])

        # Run
        samples = utils.generate_samples(covariance, [d0.ppf, d1.ppf, d2.ppf], 2)

        # Check
        assert len(samples) == 2


class TestDistribution(TestCase):

    def test_uniform_distribution(self):
        """When the values are a uniformely distributed, the Distribution recognizes it"""

        distribution = utils.Distribution(column=np.linspace(-15, 15))
        assert distribution.name == 'uniform'
        assert distribution.ppf(0.5) == 0
        assert distribution.ppf(0.01) == -distribution.ppf(0.99)

    def test_categorical_distribution(self):
        """When the keyword argument categorical=True, the Distribution behaves as such"""
        distribution = utils.Distribution(
            column=['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
            categorical=True
        )
        assert distribution.name == 'categorical'
        assert distribution.cats == ['B', 'A', 'C']
        assert distribution.estimate_args(['B', 'B', 'B', 'A', 'C']) == [0.6, 0.2, 0.2]

    def test_nan_value_logic(self):

        # Quick tests
        d0 = utils.Distribution(column=np.linspace(-15, 15))
        d1 = utils.Distribution(
            column=['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
            categorical=True
        )
        d2 = utils.Distribution(
            column=['T', 'T', 'T', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
            categorical=True
        )

        cov = np.array([[1, 0.2, 0.3], [0.2, 1, 0.5], [0.3, 0.5, 1]])
        LOGGER.debug('\nGenerated Samples:')
        LOGGER.debug(utils.generate_samples(cov, [d0.ppf, d1.ppf, d2.ppf], 2))

        # Test the np.nan values stuff
        numerical = np.array([1.0, 2.1, 3.4, np.nan, 3.4, 5.6, np.nan])
        d2 = utils.Distribution(column=np.array(numerical))
        LOGGER.debug(d2.name, d2.args)

        categorical_num = np.array([1, 2, 1, 1, 2, 1, 1, np.nan, 2, np.nan, 1])
        categorical_str = np.array(
            ['a', 'b', np.nan, 'a', np.nan, 'b'], dtype='object')

        LOGGER.debug('\nTesting NaN value logic')
        d3 = utils.Distribution(column=categorical_num, categorical=True)
        d4 = utils.Distribution(column=categorical_str, categorical=True)

        assert sum(d3.args) == 1.0
        assert sum(d4.args) == 1.0

        LOGGER.debug(zip(d3.cats, d3.args))
        LOGGER.debug(zip(d4.cats, d4.args))

        LOGGER.debug(d2.estimate_args(np.array([np.nan, np.nan, np.nan])))
        LOGGER.debug(d3.estimate_args(np.array([np.nan, np.nan, np.nan])))
        LOGGER.debug(d4.estimate_args(np.array([np.nan, np.nan, np.nan], dtype='object')))

        d5 = utils.Distribution(
            column=np.array(['a', 'b', np.nan, 'a', 'b', np.nan]),
            categorical=True
        )
        LOGGER.debug(sum(d5.args))
