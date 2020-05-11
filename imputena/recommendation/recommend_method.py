import pandas as pd


from .utils import (
    is_series, is_dataframe, is_categorical, is_numeric, contains_categorical,
    contains_only_categorical, is_temporal, has_lt_10_percent_na,
    has_gt_80_percent_cor)


def recommend_method(data=None, column=None, title_only=False):
    """Recommends an imputation method to use on a series, data frame,
    or particular column of a data frame. If data_only is True, only the
    title of the recommended method is returned, otherwise a description of
    the decision process is provided as well.

    :param data: The data for which an imputation method should be recommended.
    :type data: pandas.Series or pandas.DataFrame
    :param column: If data is a data frame, the column for which an
        imputation method should be recommended
    :type column: string, optional
    :param title_only: If true, return only the title of the imputation
        method, otherwise provide a description of the decision process as
        well.
    :type title_only: bool, default False
    :return: The title of the recommended imputation method and a
        description of the decision model if title_only is False.
    :rtype: string
    """
    # Check that data is a Series or Dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # If a column is specified while data is a Series, raise a ValueError:
    if isinstance(data, pd.Series) and column is not None:
        raise ValueError('A column can only be specified if the data is a '
                         'DataFrame.')
    # Initialize messages and method:
    messages = []
    method = None
    # Treatment if the data is a series:
    if is_series(data):
        # The data is a series.
        messages.append('The data is a series.')
        series = data
        # Check if the series contains categorical values:
        if is_categorical(series):
            messages.append('The series contains categorical values.')
            method = 'random sample imputation'
        else:
            messages.append('The series contains numerical values.')
            if is_temporal(series):
                messages.append('The series is a time series.')
                method = 'interpolation with seasonal adjustment'
            else:
                messages.append('The series is not a time series.')
                method = 'mean substitution'
    # Treatment if the data is a dataframe:
    if is_dataframe(data):
        messages.append('The data is a data frame.')
        if column is None:
            # Treatment for a whole dataframe.
            messages.append(
                'You want to apply the same method to the whole data frame.')
            if contains_categorical(data):
                # The data frame contains categorical data.
                messages.append('The data frame contains categorical data.')
                method = 'most-frequent substitution'
            else:
                # The data frame does not contain categorical data.
                messages.append(
                    'The data frame does not contain categorical data.')
                method = 'imputation using k-NN'
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
                messages.append(
                    'The column {} contains categorical values.'.format(
                        column))
                method = 'logistic regression imputation'
            else:
                # The column contains numerical values.
                messages.append(
                    'The column {} contains numerical values.'.format(
                        column))
                # Check if the column represents a time series:
                if is_temporal(series):
                    # The column represents a time series.
                    messages.append(
                        'The column {} represent a time series.'.format(
                            column))
                    method = 'interpolation with seasonal adjustment'
                else:
                    # The column does not represent a time series.
                    messages.append(
                        'The column {} does not represent a time '
                        'series.'.format(column))
                    # Check if the column contains less than 10% missing
                    # values:
                    if has_lt_10_percent_na(series):
                        # The column contains less than 10% missing values.
                        messages.append(
                            'Less than 10% of the values in the '
                            'column {} are missing.'.format(column))
                        method = 'mean substitution'
                    else:
                        # The column contains 10% or more missing values.
                        messages.append(
                            '10% or more of the values in the column'
                            '{} are missing.'.format(column))
                        # Check if the column has a correlation of more than
                        # 0.8 with any other column.
                        if has_gt_80_percent_cor(data, column):
                            # The column does have a correlation of more
                            # than 0.8 with at least one other column.
                            messages.append(
                                'The column {} has high '
                                'correlations (> 0.8) with at least one '
                                'other column.'.format(column))
                            method = 'linear regression imputation'
                        else:
                            # The column does not have a correlation of more
                            # than 0.8 with at least one other column.
                            messages.append(
                                'The column {} does not ' 
                                'have high correlations (> 0.8) with any '
                                'other column.'.format(column))
                            method = 'imputation using k-NN'
    # Create return string and return:
    res = ''
    if title_only:
        res = method
    else:
        for idx, message in enumerate(messages):
            res += str(idx + 1) + '. ' + message + '\n'
        res += 'Therefore you should apply {}.'.format(method)
    return res

