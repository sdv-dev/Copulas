import logging
from io import BytesIO

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

LOGGER = logging.getLogger(__name__)


def get_bucket(bucket_name):
    resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    return resource.Bucket(bucket_name)


def clean_dataset(data):
    numerical_col = []

    for column in data.columns:
        if (data[column].astype(int) == data[column]).all():
            numerical_col.append(column)

    data.drop(data.columns[numerical_col], axis=1, inplace=True)
    data.columns = range(data.shape[1])

    return data


def load_dataset(obj):
    body = obj.get()['Body'].read()
    data = pd.read_csv(BytesIO(body), header=None)

    return clean_dataset(data)


def get_dataset(bucket_name, dataset_name):
    bucket = get_bucket(bucket_name)
    dataset = bucket.Object(key=dataset_name)
    return load_dataset(dataset)


def get_datasets(bucket_name, limit=None):
    bucket = get_bucket(bucket_name)
    datasets = dict()
    for obj in list(bucket.objects.all()):
        dataset = load_dataset(obj)
        if not dataset.empty:
            datasets[obj.key] = dataset

        if len(datasets) >= limit:
            break

    return datasets
