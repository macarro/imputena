import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from random import shuffle

from imputena import mean_substitution, linear_regression, logistic_regression


def mice(data=None, imputations=3, regressions='available'):
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Check that the value of regressions is valid:
    if regressions not in ['available', 'complete']:
        raise ValueError(regressions + 'could not be understood.')
    # Create the list that will be returned
    imputed_datasets = []
    # Impute several times and add the results to the list:
    for _ in range(imputations):
        imputed_datasets.append(
            mice_one_imputation(data, regressions))
    # Return the imputed datasets:
    return imputed_datasets


def mice_one_imputation(data, regressions):
    # This auxiliary function always returns a copy:
    res = data.copy()
    # Save the mask of missing values:
    na_mask = pd.isna(data)
    # Compute the list of columns with missing values
    columns_with_na = []
    for column in data.columns:
        if data[column].isna().any():
            columns_with_na.append(column)
    # Shuffle the list of columns to impute:
    shuffle(columns_with_na)
    # Impute with mean substitution:
    mean_substitution(res, inplace=True)
    # Impute each column:
    for column in columns_with_na:
        if is_numeric_dtype(data[column]):
            res.loc[na_mask[column], column] = np.nan
            linear_regression(
                res, column, regressions=regressions, inplace=True)
        else:
            res.loc[na_mask[column], column] = None
            logistic_regression(
                res, column, regressions=regressions, inplace=True)
    return res

