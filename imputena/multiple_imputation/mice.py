import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from random import shuffle

from imputena import (
    mean_substitution, linear_regression, logistic_regression,
    random_sample_imputation)


def mice(data=None, imputations=3):
    """Performs multiple imputation by chained equations (MICE) on the data.
    Several (parameter imputations) linear regression imputations are
    performed on the dataset. For each one, the a random order of imputation of
    columns in generated. Then the dataset is imputed with mean
    substitution. For each column with missing data, and in the previously
    generated order, (1) the missing values imputed with the mean are set
    missing again, (2) a linear regression model is calculated based on the
    available data and (3) the predictions from the model are used to impute
    the missing values.

    :param data: The data on which to perform the MICE imputation.
    :type data: pandas.DataFrame
    :param imputations: Number of imputations to perform
    :type imputations: scalar, default 3
    :return: A list of MICE imputations performed with randomly chosen
        orders of column imputations.
    :rtype: list of pandas.DataFrame
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Create the list that will be returned
    imputed_datasets = []
    # Impute several times and add the results to the list:
    for _ in range(imputations):
        imputed_datasets.append(mice_one_imputation(data))
    # Return the imputed datasets:
    return imputed_datasets


def mice_one_imputation(data):
    """Auxiliary function that performs one MICE imputation, choosing the
    order in which the columns are imputed at random.

    :param data: The data on which to perform the imputation.
    :type data: pandas.DataFrame
    :return: The dataframe with one MICE imputation performed.
    :rtype: pandas.DataFrame
    """
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
    for column in columns_with_na:
        if is_numeric_dtype(data[column]):
            mean_substitution(res, columns=[column], inplace=True)
        else:
            random_sample_imputation(res, columns=[column], inplace=True)
    # Compute which columns are numeric in order to use them as predictors:
    numerics = [col for col in data.columns if is_numeric_dtype(data[col])]
    # Impute each column:
    for column in columns_with_na:
        if is_numeric_dtype(data[column]):
            res.loc[na_mask[column], column] = np.nan
            linear_regression(res, column, predictors=numerics, inplace=True)
        else:
            res.loc[na_mask[column], column] = None
            logistic_regression(res, column, inplace=True)
    return res

