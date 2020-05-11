import pandas as pd

from .recommend_method import recommend_method, is_series, is_dataframe
from imputena import *


def impute_by_recommended(data=None, column=None, inplace=False):
    # Check that data is a Series or Dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # If a column is specified while data is a Series, raise a ValueError:
    if isinstance(data, pd.Series) and column is not None:
        raise ValueError(
            'A column can only be specified if the data is a DataFrame.')
    # Get recommended method:
    method = recommend_method(data, column, title_only=True)
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    # Treatment if the data is a series:
    if is_series(data):
        if method == 'random sample imputation':
            res.loc[:] = random_sample_imputation(data, inplace=inplace)
        elif method == 'interpolation with seasonal adjustment':
            res.loc[:] = seasonal_interpolation(data, inplace=inplace)
        elif method == 'mean substitution':
            res.loc[:] = mean_substitution(data, inplace=inplace)
    # Treatment for a whole dataframe
    elif is_dataframe(data) and column is None:
        if method == 'most-frequent substitution':
            res.loc[:, :] = most_frequent(data)
        elif method == 'imputation using k-NN':
            res.loc[:, :] = knn(data, inplace=inplace)
    # Treatment for a column of a dataframe:
    elif is_dataframe(data) and column is not None:
        if method == 'logistic regression imputation':
            res.loc[:, :] = logistic_regression(
                data, dependent=column, inplace=inplace)
        elif method == 'interpolation with seasonal adjustment':
            res.loc[:, :] = seasonal_interpolation(
                data, columns=[column], inplace=inplace)
        elif method == 'mean substitution':
            res.loc[:, :] = mean_substitution(
                data, columns=[column], inplace=inplace)
        elif method == 'linear regression imputation':
            res.loc[:, :] = linear_regression(
                data, dependent=column, inplace=inplace)
        elif method == 'imputation using k-NN':
            res.loc[:, :] = knn(
                data, columns=[column], inplace=inplace)
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res
