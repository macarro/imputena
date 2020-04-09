import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def seasonal_interpolation(
        data=None, dec_model='multiplicative', int_method='linear',
        int_direction='both',
        columns=None,
        inplace=False):
    # Check if data is a series or dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # Raise a ValueError if columns are selected for a series:
    if isinstance(data, pd.Series) and columns is not None:
        raise ValueError('Columns can only be selected if the data is a '
                         'DataFrame.')
    # Raise a ValueError if the model is neither multiplicative nor additive:
    if dec_model not in ['multiplicative', 'additive']:
        raise ValueError(
            dec_model + 'is not a valid seasonality model.')
    if inplace:
        res = data
    else:
        res = data.copy()
    if isinstance(data, pd.Series):
        res = seasonal_interpolate_series(data, dec_model, int_method,
                                          int_direction)
    if isinstance(data, pd.DataFrame):
        if columns is None:
            columns = data.columns
        for column in columns:
            # Raise error if the column name doesn't exist in the data:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.')
            if data[column].notnull().sum() > 0:
                res[column] = seasonal_interpolate_series(
                    data[column], dec_model, int_method, int_direction)
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res


def seasonal_interpolate_series(data, dec_model, int_method, int_direction):
    res = data.copy()

    int_kwargs = {}
    int_kwargs['limit_direction'] = int_direction

    # 1. Missing data index:
    na_index = pd.isna(data)
    # 2. Interpolate NAs
    temp = data.interpolate(method=int_method, **int_kwargs)
    # 3. Decompose
    dr = seasonal_decompose(temp, model=dec_model)
    # 4. Join trend and irregular component (timeseries without
    # seasonality)
    if dec_model == 'multiplicative':
        data_no_seasonality = dr.trend * dr.resid
    if dec_model == 'additive':
        data_no_seasonality = dr.trend + dr.resid
    # 5. Fill in NA values
    data_no_seasonality[na_index] = np.nan
    # 6. Interpolate data without seasonality
    data_no_seasonality_imputed = data_no_seasonality.interpolate(
        method=int_method, **int_kwargs)
    # 7. Add seasonality
    if dec_model == 'multiplicative':
        data_imputed = data_no_seasonality_imputed * dr.seasonal
    if dec_model == 'additive':
        data_imputed = data_no_seasonality_imputed + dr.seasonal
    # 8. Merge interpolated values into original timeseries
    res[na_index] = data_imputed[na_index]
    return res

