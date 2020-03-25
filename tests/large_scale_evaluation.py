"""
Large Scale Evaluation of Copulas.

This script is a command line module that evaluates multiple MultiVariate models
from the Copulas library over a collection of real world datasets stored in an
S3 Bucket as CSV files.

Usage:

    python large_scale_evaluation.py [-h] [-v] [-o OUTPUT_PATH] [-s SAMPLE]
                                     [-r MAX_ROWS] [-c MAX_COLUMNS]
                                     [-m MODEL [MODEL ...]]
                                     [datasets [datasets ...]]

    positional arguments:
      datasets              Name of the datasets/s to test.

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         Be verbose. Use -vv for increased verbosity.
      -o OUTPUT_PATH, --output-path OUTPUT_PATH
                            Path to the CSV file where the report will be dumped
      -s SAMPLE, --sample SAMPLE
                            Limit the test to a number of datasets (sampled
                            randomly) specified by SAMPLE.
      -r MAX_ROWS, --max-rows MAX_ROWS
                            Limit the number of rows per dataset.
      -c MAX_COLUMNS, --max-columns MAX_COLUMNS
                            Limit the number of columns per dataset.
      -m MODEL [MODEL ...], --model MODEL [MODEL ...]
                            Name of the model to test. Can be passed multiple
                            times to evaluate more than one model.
"""
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

LOGGER = logging.getLogger(__name__)

BUCKET_NAME = 'atm-data'  # Bucket where the datasets are stored
DATA_URL = 'http://{}.s3.amazonaws.com/'.format(BUCKET_NAME)

AVAILABLE_MODELS = {
    'GaussianMultivariate(GaussianUnivariate)': GaussianMultivariate(GaussianUnivariate),
    'GaussianMultivariate()': GaussianMultivariate(),
    'VineCopula("center")': VineCopula('center'),
    'VineCopula("direct")': VineCopula('direct'),
    'VineCopula("regular")': VineCopula('regular')
}
OUTPUT_COLUMNS = [
    'model_name',
    'dataset_name',
    'num_columns',
    'num_rows',
    'elapsed_time',
    'score',
    'error_message',
]


def get_available_datasets_list():
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    available_datasets = [
        obj['Key']
        for obj in client.list_objects(Bucket=BUCKET_NAME)['Contents']
        if obj['Key'] != 'index.html'
    ]

    return available_datasets


def get_dataset_url(name):
    if not name.endswith('.csv'):
        name = name + '.csv'

    return urljoin(DATA_URL, name)


def load_data(dataset_name, max_rows, max_columns):
    LOGGER.debug('Loading dataset %s (max_rows: %s, max_columns: %s)',
                 dataset_name, max_rows, max_columns)
    dataset_url = get_dataset_url(dataset_name)
    data = pd.read_csv(dataset_url, nrows=max_rows)
    if max_columns:
        data = data[data.columns[:max_columns]]

    return data


def evaluate_model_dataset(model_name, dataset_name, max_rows, max_columns):
    data = load_data(dataset_name, max_rows, max_columns)
    start = datetime.utcnow()

    LOGGER.info('Testing dataset %s (shape: %s)', dataset_name, data.shape)
    LOGGER.debug('dtypes for dataset %s:\n%s', dataset_name, data.dtypes)

    error_message = None
    score = None
    try:
        model = AVAILABLE_MODELS.get(model_name, model_name)
        instance = get_instance(model)
        LOGGER.info('Fitting dataset %s (shape: %s)', dataset_name, data.shape)
        instance.fit(data)

        LOGGER.info('Sampling %s rows for dataset %s', len(data), dataset_name)
        sampled = instance.sample(len(data))
        assert sampled.shape == data.shape

        try:
            LOGGER.info('Computing PDF for dataset %s', dataset_name)
            pdf = instance.pdf(sampled)
            assert (0 <= pdf).all()

            LOGGER.info('Computing CDF for dataset %s', dataset_name)
            cdf = instance.cdf(sampled)
            assert (0 <= cdf).all() and (cdf <= 1).all()
        except NotImplementedError:
            pass

        LOGGER.info('Evaluating scores for dataset %s', dataset_name)
        scores = []
        for column in data.columns:
            scores.append(ks_2samp(sampled[column].values, data[column].values))

        score = np.mean(scores)
        LOGGER.info("Dataset %s score: %s", dataset_name, score)

    except Exception as ex:
        error_message = '{}: {}'.format(ex.__class__.__name__, ex)
        LOGGER.exception("Dataset %s failed: %s", dataset_name, error_message)

    elapsed_time = datetime.utcnow() - start

    return {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'elapsed_time': elapsed_time,
        'error_message': error_message,
        'score': score,
        'num_columns': len(data.columns),
        'num_rows': len(data)
    }


def run_evaluation(model_names, dataset_names, max_rows, max_columns):
    start = datetime.utcnow()
    results = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            result = evaluate_model_dataset(model_name, dataset_name, max_rows, max_columns)
            results.append(result)

        elapsed_time = datetime.utcnow() - start
        LOGGER.info('%s datasets tested using model %s in %s',
                    len(dataset_names), model_name, elapsed_time)

    elapsed_time = datetime.utcnow() - start
    LOGGER.info('%s datasets tested %s models in %s',
                len(dataset_names), len(model_names), elapsed_time)

    return pd.DataFrame(results, columns=OUTPUT_COLUMNS)


def _valid_model(name):
    if name not in AVAILABLE_MODELS:
        msg = 'Unknown model: {}\nValid models are: {}'.format(name, list(AVAILABLE_MODELS.keys()))
        raise argparse.ArgumentTypeError(msg)

    return name


def _get_parser():
    # Parser
    parser = argparse.ArgumentParser(description='Large scale Copulas evaluation')

    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')
    parser.add_argument('-o', '--output-path', type=str, required=False,
                        help='Path to the CSV file where the report will be dumped')
    parser.add_argument('-s', '--sample', type=int,
                        help=(
                            'Limit the test to a number of datasets (sampled randomly)'
                            ' specified by SAMPLE.'
                        ))
    parser.add_argument('-r', '--max-rows', type=int,
                        help='Limit the number of rows per dataset.')
    parser.add_argument('-c', '--max-columns', type=int,
                        help='Limit the number of columns per dataset.')
    parser.add_argument('-m', '--model', nargs='+', type=_valid_model,
                        help=(
                            'Name of the model to test. Can be passed multiple '
                            'times to evaluate more than one model.'
                        ))
    parser.add_argument('datasets', nargs='*',
                        help=(
                            'Name of the datasets/s to test. If no names are given '
                            'all the available datasets are tested.'
                        ))

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    log_level = (3 - args.verbose) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=fmt)

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = get_available_datasets_list()
        if args.sample:
            dataset_names = random.sample(dataset_names, args.sample)

    model_names = args.model or list(AVAILABLE_MODELS.keys())
    LOGGER.info("Testing datasets %s on models %s", dataset_names, model_names)

    results = run_evaluation(model_names, dataset_names, args.max_rows, args.max_columns)

    print(tabulate.tabulate(
        results,
        tablefmt='github',
        headers=results.columns,
        showindex=False
    ))

    if args.output_path:
        LOGGER.info('Saving report to %s', args.output_path)
        results.to_csv(args.output_path)


if __name__ == '__main__':
    main()
