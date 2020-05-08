import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def seasonal_interpolation(
        data=None, dec_model='multiplicative', int_method='linear',
        int_direction='both', columns=None, inplace=False):
    """Performs interpolation with seasonal adjustment on a time series or a
    data frame containing time series. First, the time series gets
    decomposed according to the decomposition model (additive or
    multiplicative). Then, the missing values are interpolated using the
    interpolation method (linear, cubic or quadratic) on a series consisting of
    only the trend and irregular components. Finally, the seasonality is
    added back to the series.

    :param data: The data on which to perform the seasonal interpolation.
    :type data: pandas.Series or pandas.DataFrame
    :param dec_model: The decomposition model to use.
    :type dec_model: {'multiplicative', 'additive'}, default 'multiplicative'
    :param int_method: The interpolation model to use.
    :type int_method: {'linear', 'quadratic', 'cubic'}
    :param int_direction: Direction in which to interpolate values when the
        interpolation method is linear.
    :type int_direction: {'forward', 'backward', 'both'}, default 'both'
    :param columns: Columns on which to apply the operation.
    :type columns: array-like, optional
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The series or dataframe with NA values interpolated, or
        None if inplace=True.
    :rtype: pandas.Series, pandas.DataFrame, or None
    :raises: TypeError, ValueError
    """
    # Check that data is a Series or DataFrame:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # Raise a ValueError if columns are selected for a Series:
    if isinstance(data, pd.Series) and columns is not None:
        raise ValueError(
            'Columns can only be selected if the data is a DataFrame.')
    # Check if dec_model has a valid value:
    if dec_model not in ['multiplicative', 'additive']:
        raise ValueError(dec_model + 'is not a supported decomposition model.')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    # Treatment if the data is a Series:
    if isinstance(data, pd.Series):
        # The operation is only applied if the column contains non-NA values:
        if data.notnull().sum() > 0:
            res[:] = seasonal_interpolate_series(
                data, dec_model, int_method, int_direction)
    # Treatment if the data is a DataFrame:
    if isinstance(data, pd.DataFrame):
        # If no columns are given, apply the operations to all columns of
        # the dataframe:
        if columns is None:
            columns = data.columns
        for column in columns:
            # Raise error if the column name doesn't exist in the data:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.')
            # The operation is only applied if the column contains non-NA
            # values:
            if data[column].notnull().sum() > 0:
                res[column] = seasonal_interpolate_series(
                    data[column], dec_model, int_method, int_direction)
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res


def seasonal_interpolate_series(data, dec_model, int_method, int_direction):
    """Auxiliary function that interpolates a series with seasonal
    adjustment. It always returns a copy.

    :param data: The series on which to perform the operation.
    :type data: pandas.Series
    :param dec_model: The decomposition model to use. Passed to
        statsmodels.tsa.seasonal.seasonal_decompose().
    :type dec_model: {'multiplicative', 'additive'}
    :param int_method: The interpolation model to use. Passed to
        pandas.DataFrame.interpolate()
    :type int_method: {'linear', 'quadratic', 'cubic'}
    :param int_direction: Direction in which to interpolate values. Passed to
        pandas.DataFrame.interpolate()
    :type int_direction: {'forward', 'backward', 'both'}
    :return: The data interpolated with seasonal adjustment.
    :rtype: pandas.Series
    """
    # This function aways returns a copy. The parent function takes care of
    # assigning its results to the same series or data frame if the
    # operation is to be made inplace.
    res = data.copy()
    # kwargs for the pandas interpolate() function:
    int_kwargs = {'limit_direction': int_direction}
    # 1. Missing data index:
    na_index = pd.isna(data)
    # 2. Interpolate NAs:
    temp = data.interpolate(method=int_method, **int_kwargs)
    # 3. Decompose:
    dr = seasonal_decompose(temp, model=dec_model)
    # 4. Join trend and irregular component (timeseries without seasonality):
    if dec_model == 'multiplicative':
        data_no_seasonality = dr.trend * dr.resid
    if dec_model == 'additive':
        data_no_seasonality = dr.trend + dr.resid
    # 5. Fill in NA values:
    data_no_seasonality[na_index] = np.nan
    # 6. Interpolate data without seasonality:
    data_no_seasonality_imputed = data_no_seasonality.interpolate(
        method=int_method, **int_kwargs)
    # 7. Add back seasonality:
    if dec_model == 'multiplicative':
        data_imputed = data_no_seasonality_imputed * dr.seasonal
    if dec_model == 'additive':
        data_imputed = data_no_seasonality_imputed + dr.seasonal
    # 8. Merge interpolated values into original timeseries:
    res[na_index] = data_imputed[na_index]
    # Return the seasonally interpolated series:
    return res
