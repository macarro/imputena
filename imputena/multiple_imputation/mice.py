import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from random import shuffle

from imputena import mean_substitution, linear_regression, logistic_regression


def mice(data=None, imputations=3, regressions='available'):
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
    :param regressions: If 'available': Impute missing values by modeling a
        regression based on all available predictors if some predictors have
        missing values themselves. If 'complete': Only impute with a
        regression model based on all predictors and leave missing values in
        rows in which some predictor value is missing itself unimputed.
    :type regressions: {'available', 'complete'}, default 'available'
    :return: A list of MICE imputations performed with randomly chosen
        orders of column imputations.
    :rtype: list of pandas.DataFrame
    :raises: TypeError, ValueError
    """
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
    """Auxiliary function that performs one MICE imputation, choosing the
    order in which the columns are imputed at random.

    :param data: The data on which to perform the imputation.
    :type data: pandas.DataFrame
    :param regressions: If 'available': Impute missing values by modeling a
        regression based on all available predictors if some predictors have
        missing values themselves. If 'complete': Only impute with a
        regression model based on all predictors and leave missing values in
        rows in which some predictor value is missing itself unimputed.
    :type regressions: {'available', 'complete'}
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

