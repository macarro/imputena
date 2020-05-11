"""Auxiliary functions used by several recommendation and aplicability
functions.
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype


def is_series(data):
    """Auxiliary function that checks whether the data is a series.

    :param data: The data to check
    :type data: any
    :return: Whether the data is a series
    :rtype: bool
    """
    return isinstance(data, pd.Series)


def is_dataframe(data):
    """Auxiliary function that checks whether the data is a data frame.

    :param data: The data to check
    :type data: any
    :return: Whether the data is a data frame
    :rtype: bool
    """
    return isinstance(data, pd.DataFrame)


def is_numeric(series):
    """Auxiliary function that checks whether a series contains numerical
    values.

    :param series: The series to check
    :type series: pandas.Series
    :return: Whether the series contains numerical values
    :rtype: bool
    """
    return is_numeric_dtype(series)


def is_categorical(series):
    """Auxiliary function that checks whether a series contains categorical
    values.

    :param series: The series to check
    :type series: pandas.Series
    :return: Whether the series contains categorical values
    :rtype: bool
    """
    return not is_numeric(series)


def contains_categorical(dataframe):
    """Auxiliary function that checks whether a data frame contains categorical
    values in at least one column.

    :param dataframe: The data frame to check
    :type dataframe: pandas.DataFrame
    :return: Whether the data frame contains categorical values in at least
        one column.
    :rtype: bool
    """
    contains = False
    for column in dataframe.columns:
        if is_categorical(dataframe[column]):
            contains = True
            break
    return contains


def contains_only_categorical(dataframe):
    """Auxiliary function that checks whether a data frame contains categorical
    values in all columns.

    :param dataframe: The data frame to check
    :type dataframe: pandas.DataFrame
    :return: Whether the data frame contains categorical values in at least
        one column.
    :rtype: bool
    """
    contains_only = True
    for column in dataframe.columns:
        if not is_categorical(dataframe[column]):
            contains_only = False
            break
    return contains_only


def is_temporal(data):
    """Auxiliary function that checks whether a series or dataframe has a
    datetime index is therefore temporal.

    :param data: The data to check
    :type data: pandas.Series or pandas.DataFrame
    :return: Whether the data has a datetime index
    :rtype: bool
    """
    return isinstance(data.index, pd.DatetimeIndex)


def has_gt_80_percent_cor(data, column):
    """"Auxiliary function that checks whether a specified column of a data
    frame has a correlation of more than 0.8 with at least one other column.

    :param data: The data frame on which to perform the check
    :type data: pandas.DataFrame
    :param column: The column of the data frame to check
    :type column: string
    :return: Whether the column has a correlation of more than 0.8 with at
        least one other column.
    :rtype: bool
    """
    return data.corr()[column].sort_values(ascending=False)[1] > 0.8


def has_lt_10_percent_na(series):
    """Auxiliary function that checks whether less than 10% of a series's
    values are missing

    :param series: The series to check
    :type series: pandas.Series
    :return: Whether less than 10% of the series's values are missing
    :rtype: bool
    """
    return series.isna().sum() / len(series.index) < .1