"""SDMetrics utils to be used across all the project."""

from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def nested_attrs_meta(nested):
    """Metaclass factory that defines a Metaclass with a dynamic attribute name."""

    class Metaclass(type):
        """Metaclass which pulls the attributes from a nested object using properties."""

        def __getattr__(cls, attr):
            """If cls does not have the attribute, try to get it from the nested object."""
            nested_obj = getattr(cls, nested)
            if hasattr(nested_obj, attr):
                return getattr(nested_obj, attr)

            raise AttributeError(f"type object '{cls.__name__}' has no attribute '{attr}'")

        @property
        def name(cls):
            return getattr(cls, nested).name

        @property
        def goal(cls):
            return getattr(cls, nested).goal

        @property
        def max_value(cls):
            return getattr(cls, nested).max_value

        @property
        def min_value(cls):
            return getattr(cls, nested).min_value

    return Metaclass


def get_frequencies(real, synthetic):
    """Get percentual frequencies for each possible real categorical value.

    Given two iterators containing categorical data, this transforms it into
    observed/expected frequencies which can be used for statistical tests. It
    adds a regularization term to handle cases where the synthetic data contains
    values that don't exist in the real data.

    Args:
        real (list):
            A list of hashable objects.
        synthetic (list):
            A list of hashable objects.

    Yields:
        tuble[list, list]:
            The observed and expected frequencies (as a percent).
    """
    f_obs, f_exp = [], []
    real, synthetic = Counter(real), Counter(synthetic)
    for value in synthetic:
        if value not in real:
            real[value] += 1e-6  # Regularization to prevent NaN.

    for value in real:
        f_obs.append(synthetic[value] / sum(synthetic.values()))  # noqa: PD011
        f_exp.append(real[value] / sum(real.values()))  # noqa: PD011

    return f_obs, f_exp


def get_missing_percentage(data_column):
    """Compute the missing value percentage of a column.

    Args:
        data_column (pandas.Series):
            The data of the desired column.

    Returns:
        pandas.Series:
            Percentage of missing values inside the column.
    """
    return round((data_column.isna().sum() / len(data_column)) * 100, 2)


def get_cardinality_distribution(parent_column, child_column):
    """Compute the cardinality distribution of the (parent, child) pairing.

    Args:
        parent_column (pandas.Series):
            The parent column.
        child_column (pandas.Series):
            The child column.

    Returns:
        pandas.Series:
            The cardinality distribution.
    """
    child_df = pd.DataFrame({'child_counts': child_column.value_counts()})
    cardinality_df = pd.DataFrame({'parent': parent_column}).join(
        child_df, on='parent').fillna(0)

    return cardinality_df['child_counts']


def is_datetime(data):
    """Determine if the input is a datetime type or not.

    Args:
        data (pandas.DataFrame, int or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    return (
        pd.api.types.is_datetime64_any_dtype(data)
        or isinstance(data, pd.Timestamp)
        or isinstance(data, datetime)
    )


class HyperTransformer():
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a set of transforms to transform one or
    more columns based on each column's data type.
    """

    column_transforms = {}
    column_kind = {}

    def fit(self, data):
        """Fit the HyperTransformer to the given data.

        Args:
            data (pandas.DataFrame):
                The data to transform.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            kind = data[field].dropna().infer_objects().dtype.kind
            self.column_kind[field] = kind

            if kind == 'i' or kind == 'f':
                # Numerical column.
                self.column_transforms[field] = {'mean': data[field].mean()}
            elif kind == 'b':
                # Boolean column.
                numeric = pd.to_numeric(data[field], errors='coerce').astype(float)
                self.column_transforms[field] = {'mode': numeric.mode().iloc[0]}
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                enc = OneHotEncoder()
                enc.fit(col_data)
                self.column_transforms[field] = {'one_hot_encoder': enc}
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(
                    data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                self.column_transforms[field] = {'mean': pd.Series(integers).mean()}

    def transform(self, data):
        """Transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            transform_info = self.column_transforms[field]

            kind = self.column_kind[field]
            if kind == 'i' or kind == 'f':
                # Numerical column.
                data[field] = data[field].fillna(transform_info['mean'])
            elif kind == 'b':
                # Boolean column.
                data[field] = pd.to_numeric(data[field], errors='coerce').astype(float)
                data[field] = data[field].fillna(transform_info['mode'])
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                out = transform_info['one_hot_encoder'].transform(col_data).toarray()
                transformed = pd.DataFrame(
                    out, columns=[f'value{i}' for i in range(np.shape(out)[1])])
                data = data.drop(columns=[field])
                data = pd.concat([data, transformed.set_index(data.index)], axis=1)
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(
                    data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                data[field] = pd.Series(integers)
                data[field] = data[field].fillna(transform_info['mean'])

        return data

    def fit_transform(self, data):
        """Fit and transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        self.fit(data)
        return self.transform(data)


def get_columns_from_metadata(metadata):
    """Get the column info from a metadata dict.

    Args:
        metadata (dict):
            The metadata dict.

    Returns:
        dict:
            The columns metadata.
    """
    return metadata.get('columns', {})


def get_type_from_column_meta(column_metadata):
    """Get the type of a given column from the column metadata.

    Args:
        column_metadata (dict):
            The column metadata.

    Returns:
        string:
            The column type.
    """
    return column_metadata.get('sdtype', '')


def get_alternate_keys(metadata):
    """Get the alternate keys from a metadata dict.

    Args:
        metadata (dict):
            The metadata dict.

    Returns:
        list:
            The list of alternate keys.
    """
    alternate_keys = []
    for alternate_key in metadata.get('alternate_keys', []):
        if isinstance(alternate_key, list):
            alternate_keys.extend(alternate_key)
        else:
            alternate_keys.append(alternate_key)

    return alternate_keys


def strip_characters(list_character, a_string):
    """Strip characters from a column name.

    Args:
        list_character (list):
            The list of characters to strip.
        a_string (string):
            The string to be stripped.

    Returns:
        string:
            The string with the characters stripped.
    """
    result = a_string
    for character in list_character:
        if character in result:
            result = result.replace(character, '')

    return result
