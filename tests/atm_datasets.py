import argparse
import logging
import random
from datetime import datetime
from urllib.parse import urljoin

import boto3
import numpy as np
import pandas as pd
import tabulate
from botocore import UNSIGNED
from botocore.client import Config
from scipy.stats import ks_2samp

from copulas import get_instance
from copulas.multivariate import GaussianMultivariate, VineCopula
from copulas.univariate import GaussianUnivariate

ATM_DATA_URL = 'http://atm-data.s3.amazonaws.com/'
LOGGER = logging.getLogger(__name__)


MODELS = {
    'GaussianMultivariate(GaussianUnivariate)': GaussianMultivariate(GaussianUnivariate),
    'GaussianMultivariate()': GaussianMultivariate(),
    'VineCopula("center")': VineCopula('center'),
    'VineCopula("direct")': VineCopula('direct'),
    'VineCopula("regular")': VineCopula('regular')
}


def get_available_datasets():
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    available_datasets = [
        obj['Key']
        for obj in client.list_objects(Bucket='atm-data')['Contents']
        if obj['Key'] != 'index.html'
    ]

    return available_datasets


def get_atm_dataset_url(name):
    if not name.endswith('.csv'):
        name = name + '.csv'

    return urljoin(ATM_DATA_URL, name)


def load_data(dataset, max_rows):
    LOGGER.debug('Loading dataset %s (max_rows: %s)', dataset, max_rows)
    url = get_atm_dataset_url(dataset)
    return pd.read_csv(url, nrows=max_rows)


def test_dataset(model, dataset, max_rows):
    data = load_data(dataset, max_rows)
    start = datetime.utcnow()

    LOGGER.info('Testing dataset %s (shape: %s)', dataset, data.shape)
    LOGGER.debug('dtypes for dataset %s:\n%s', dataset, data.dtypes)

    error = None
    score = None
    try:
        instance = get_instance(MODELS.get(model, model))
        LOGGER.info('Fitting dataset %s (shape: %s)', dataset, data.shape)
        instance.fit(data)

        LOGGER.info('Sampling %s rows for dataset %s', len(data), dataset)
        sampled = instance.sample(len(data))
        assert sampled.shape == data.shape

        LOGGER.info('Evaluating scores for dataset %s', dataset)
        scores = []
        for column in data.columns:
            scores.append(ks_2samp(sampled[column].values, data[column].values))

        score = np.mean(scores)
        LOGGER.info("Dataset %s score: %s", dataset, score)

    except Exception as ex:
        error = str(ex)
        LOGGER.error("Dataset %s failed: %s", dataset, ex)

    elapsed = datetime.utcnow() - start

    return {
        'model': model,
        'dataset': dataset,
        'elapsed': elapsed,
        'error': error,
        'score': score,
        'columns': len(data.columns),
        'rows': len(data)
    }


COLUMNS = [
    'model',
    'dataset',
    'columns',
    'rows',
    'elapsed',
    'score',
    'error',
]


def run_test(models, datasets, max_rows):
    start = datetime.utcnow()
    results = []
    for model in models:
        for dataset in datasets:
            results.append(test_dataset(model, dataset, max_rows))

        elapsed = datetime.utcnow() - start
        LOGGER.info('%s datasets tested using model %s in %s', len(datasets), model, elapsed)

    elapsed = datetime.utcnow() - start
    LOGGER.info('%s datasets tested %s models in %s', len(datasets), len(models), elapsed)

    return pd.DataFrame(results, columns=COLUMNS)


def logging_setup(verbosity=1, logfile=None, logger_name=None, stdout=True):
    logger = logging.getLogger(logger_name)
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout or not logfile:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def _get_parser():
    # Parser
    parser = argparse.ArgumentParser(description='ATM Datasets Test')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')
    parser.add_argument('-o', '--output', type=str, required=False,
                        help='Path to the CSV file where the report will be dumped')
    parser.add_argument('-s', '--sample', type=int,
                        help='Limit the test to a sample of datasets for the given size.')
    parser.add_argument('-r', '--max-rows', type=int,
                        help='Limit the number of rows per dataset.')
    parser.add_argument('-m', '--models', nargs='+', default=MODELS.keys(),
                        help='Limit the number of rows per dataset.')
    parser.add_argument('datasets', nargs='*', help='Name of the datasets/s to test.')

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    logging_setup(args.verbose)

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = get_available_datasets()
        if args.sample:
            datasets = random.sample(datasets, args.sample)

    LOGGER.info("Testing datasets %s", datasets)

    results = run_test(args.models, datasets, args.max_rows)

    print(tabulate.tabulate(
        results,
        tablefmt='github',
        headers=results.columns
    ))

    if args.output:
        LOGGER.info('Saving report to %s', args.output)
        results.to_csv(args.output)


if __name__ == '__main__':
    main()
