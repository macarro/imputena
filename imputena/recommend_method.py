import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.api.types import is_numeric_dtype


def recommend_method(data=None, column=None, title_only=False):
    # Check that data is a Series or Dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # If a column is specified while data is a Series, raise a ValueError:
    if isinstance(data, pd.Series) and column is not None:
        raise ValueError('A column can only be specified if the data is a '
                         'DataFrame.')
    # Initialize the return message:
    message = ''
    res = None
    res = None
    if is_series(data):
        message += 'The data is a series.'
        series = data
        if is_categorical(series):
            message += '\nThe series contains categorical values.'
            res = create_message(
                message, 'most-frequent substitution', title_only)
        else:
            message += '\nThe series contains numerical values.'
            if is_timeseries(series):
                message += '\nThe series is a time series.'
                res = create_message(
                    message, 'interpolation with seasonal adjustment',
                    title_only)
            else:
                message += '\nThe series is not a time series.'
                res = create_message(
                    message, 'mean substitution', title_only)
    if is_dataframe(data):
        message += 'The data is a data frame.'
        if column is None:
            # Treatment for a whole dataframe.
            message += '\nYou want to apply the same method to the whole ' \
                'data frame.'
            # Check if any column is numerical:
            contains_categorical = False
            for column in data.columns:
                if is_categorical(data[column]):
                    contains_categorical = True
                    break
            if contains_categorical:
                # The data frame contains categorical data.
                message += '\nThe data frame contains categorical data.'
                res = create_message(
                    message, 'most-frequent substitution', title_only)
            else:
                # The data frame does not contain categorical data.
                message += '\nThe data frame does not contain categorical ' \
                    'data.'
                res = create_message(
                    message, 'imputation using k-NN', title_only)
        else:
            # Treatment for a specific column of a dataframe
            # Check if column is actually a column of data:
            if column in data.columns:
                series = data[column]
            else:
                raise ValueError(column + 'is not a column of the data.')
            # Check if the column contains categorical values:
            if is_categorical(series):
                # The column contains categorical values.
                message += '\nThe column ' + column + ' contains categorical' \
                    ' values.'
                res = create_message(
                    message, 'logistic regression imputation',
                    title_only)
            else:
                # The column contains numerical values.
                message += '\nThe column ' + column + ' contains numerical' \
                    ' values.'
                # Check if the column represents a time series:
                if is_timeseries(series):
                    # The column represents a time series.
                    message += '\nThe column ' + column + ' represent a time' \
                        ' series.'
                    res = create_message(
                        message, 'interpolation with seasonal adjustment',
                        title_only)
                else:
                    # The column does not represent a time series.
                    message += '\nThe column ' + column + ' does not' \
                         ' represent a time series.'
                    # Check if the column contains less than 10% missing
                    # values:
                    if has_lt_10_percent_na(series):
                        # The column contains less than 10% missing values.
                        message += '\nLess than 10% of the values in the ' \
                            'column ' + column + ' are missing.'
                        res = create_message(
                            message, 'mean substitution', title_only)
                    else:
                        # The column contains 10% or more missing values.
                        message += '\n10% or more of the values in the ' \
                                   'column ' + column + ' are missing.'
                        # Check if the column has a correlation of more than
                        # 0.8 with any other column.
                        if has_gt_80_percent_cor(data, column):
                            # The column does have a correlation of more
                            # than 0.8 with at least one other column.
                            message += '\nThe column' + column + ' has high ' \
                                'correlations (> 0.8) with at least one ' \
                                'other column.'
                            res = create_message(
                                message, 'linear regression imputation',
                                title_only)
                        else:
                            # The column does not have a correlation of more
                            # than 0.8 with at least one other column.
                            message += '\nThe column' + column + ' does not ' \
                                'have high correlations (> 0.8) with any ' \
                                'other column.'
                            res = create_message(
                                message, 'imputation using k-NN', title_only)
    return res


def is_series(data):
    return isinstance(data, pd.Series)


def is_dataframe(data):
    return isinstance(data, pd.DataFrame)


def is_numeric(series):
    return is_numeric_dtype(series)


def is_categorical(series):
    return not is_numeric(series)


def is_timeseries(series):
    return isinstance(series.index, pd.DatetimeIndex)


def has_gt_80_percent_cor(data, column):
    return data.corr()[column].sort_values(ascending=False)[1] > 0.8


def has_lt_10_percent_na(series):
    return series.isna().sum() / len(series.index) < .1


def create_message(message, title, title_only):
    res = ''
    if title_only:
        res = title
    else:
        res = message + '\nTherefore you should use ' + title + '.'
    return res
