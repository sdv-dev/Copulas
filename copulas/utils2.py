"""Report utility methods."""

import copy
import itertools
import warnings

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

from copulas.utils import (
    get_alternate_keys, get_columns_from_metadata, get_type_from_column_meta, is_datetime)

CONTINUOUS_SDTYPES = ['numerical', 'datetime']
DISCRETE_SDTYPES = ['categorical', 'boolean']


class PlotConfig:
    """Custom plot settings for visualizations."""

    GREEN = '#36B37E'
    RED = '#FF0000'
    ORANGE = '#F16141'
    DATACEBO_DARK = '#000036'
    DATACEBO_GREEN = '#01E0C9'
    DATACEBO_BLUE = '#03AFF1'
    BACKGROUND_COLOR = '#F5F5F8'
    FONT_SIZE = 18


def convert_to_datetime(column_data, datetime_format=None):
    """Convert a column data to pandas datetime.

    Args:
        column_data (pandas.Series):
            The column data
        format (str):
            Optional string format of datetime. If ``None``, will attempt to infer the datetime
            format from the column data. Defaults to ``None``.

    Returns:
        pandas.Series:
            The converted column data.
    """
    if is_datetime(column_data):
        return column_data

    if datetime_format is None:
        datetime_format = _guess_datetime_format_for_array(column_data.astype(str).to_numpy())

    return pd.to_datetime(column_data, format=datetime_format)


def convert_datetime_columns(real_column, synthetic_column, col_metadata):
    """Convert a real and a synthetic column to pandas datetime.

    Args:
        real_data (pandas.Series):
            The real column data
        synthetic_column (pandas.Series):
            The synthetic column data
        col_metadata:
            The metadata associated with the column

    Returns:
        (pandas.Series, pandas.Series):
            The converted real and synthetic column data.
    """
    datetime_format = col_metadata.get('format') or col_metadata.get('datetime_format')
    return (convert_to_datetime(real_column, datetime_format),
            convert_to_datetime(synthetic_column, datetime_format))


def discretize_table_data(real_data, synthetic_data, metadata):
    """Create a copy of the real and synthetic data with discretized data.

    Convert numerical and datetime columns to discrete values, and label them
    as categorical.

    Args:
        real_data (pandas.DataFrame):
            The real data.
        synthetic_data (pandas.DataFrame):
            The synthetic data.
        metadata (dict)
            The metadata.

    Returns:
        (pandas.DataFrame, pandas.DataFrame, dict):
            The binned real and synthetic data, and the updated metadata.
    """
    binned_real = real_data.copy()
    binned_synthetic = synthetic_data.copy()
    binned_metadata = copy.deepcopy(metadata)

    for column_name, column_meta in get_columns_from_metadata(metadata).items():
        sdtype = get_type_from_column_meta(column_meta)

        if sdtype in ('numerical', 'datetime'):
            real_col = real_data[column_name]
            synthetic_col = synthetic_data[column_name]
            if sdtype == 'datetime':
                datetime_format = column_meta.get('format') or column_meta.get('datetime_format')
                if real_col.dtype == 'O' and datetime_format:
                    real_col = pd.to_datetime(real_col, format=datetime_format)
                    synthetic_col = pd.to_datetime(synthetic_col, format=datetime_format)

                real_col = pd.to_numeric(real_col)
                synthetic_col = pd.to_numeric(synthetic_col)

            bin_edges = np.histogram_bin_edges(real_col.dropna())
            binned_real_col = np.digitize(real_col, bins=bin_edges)
            binned_synthetic_col = np.digitize(synthetic_col, bins=bin_edges)

            binned_real[column_name] = binned_real_col
            binned_synthetic[column_name] = binned_synthetic_col
            get_columns_from_metadata(binned_metadata)[column_name] = {'sdtype': 'categorical'}

    return binned_real, binned_synthetic, binned_metadata


def _get_non_id_columns(metadata, binned_metadata):
    valid_sdtypes = ['numerical', 'categorical', 'boolean', 'datetime']
    alternate_keys = get_alternate_keys(metadata)
    non_id_columns = []
    for column, column_meta in get_columns_from_metadata(binned_metadata).items():
        is_key = column == metadata.get('primary_key', '') or column in alternate_keys
        if get_type_from_column_meta(column_meta) in valid_sdtypes and not is_key:
            non_id_columns.append(column)

    return non_id_columns


def discretize_and_apply_metric(real_data, synthetic_data, metadata, metric, keys_to_skip=[]):
    """Discretize the data and apply the given metric.

    Args:
        real_data (pandas.DataFrame):
            The real data.
        synthetic_data (pandas.DataFrame):
            The synthetic data.
        metadata (dict)
            The metadata.
        metric (sdmetrics.single_table.MultiColumnPairMetric):
            The column pair metric to apply.
        keys_to_skip (list[tuple(str)] or None):
            A list of keys for which to skip computing the metric.

    Returns:
        dict:
            The metric results.
    """
    metric_results = {}

    binned_real, binned_synthetic, binned_metadata = discretize_table_data(
        real_data, synthetic_data, metadata)

    non_id_cols = _get_non_id_columns(metadata, binned_metadata)
    for columns in itertools.combinations(non_id_cols, r=2):
        sorted_columns = tuple(sorted(columns))
        if (
            sorted_columns not in keys_to_skip and
            (sorted_columns[1], sorted_columns[0]) not in keys_to_skip
        ):
            result = metric.column_pairs_metric.compute_breakdown(
                binned_real[list(sorted_columns)],
                binned_synthetic[list(sorted_columns)],
            )
            metric_results[sorted_columns] = result
            metric_results[sorted_columns] = result

    return metric_results


def aggregate_metric_results(metric_results):
    """Aggregate the scores and errors in a metric results mapping.

    Args:
        metric_results (dict):
            The metric results to aggregate.

    Returns:
        (float, int):
            The average of the metric scores, and the number of errors.
    """
    if len(metric_results) == 0:
        return np.nan, 0

    metric_scores = []
    num_errors = 0

    for _, breakdown in metric_results.items():
        metric_score = breakdown.get('score', np.nan)
        if not np.isnan(metric_score):
            metric_scores.append(metric_score)
        if 'error' in breakdown:
            num_errors += 1

    return np.mean(metric_scores), num_errors


def _validate_categorical_values(real_data, synthetic_data, metadata, table=None):
    """Get categorical values found in synthetic data but not real data for all columns.

    Args:
        real_data (pd.DataFrame):
            The real data.
        synthetic_data (pd.DataFrame):
            The synthetic data.
        metadata (dict):
            The metadata.
        table (str, optional):
            The name of the current table, if one exists
    """
    if table:
        warning_format = ('Unexpected values ({values}) in column "{column}" '
                          f'and table "{table}"')
    else:
        warning_format = 'Unexpected values ({values}) in column "{column}"'

    columns = get_columns_from_metadata(metadata)
    for column, column_meta in columns.items():
        column_type = get_type_from_column_meta(column_meta)
        if column_type == 'categorical':
            extra_categories = [
                value for value in synthetic_data[column].unique()
                if value not in real_data[column].unique()
            ]
            if extra_categories:
                value_list = '", "'.join(str(value) for value in extra_categories[:5])
                values = f'"{value_list}" + more' if len(
                    extra_categories) > 5 else f'"{value_list}"'
                warnings.warn(warning_format.format(values=values, column=column))
