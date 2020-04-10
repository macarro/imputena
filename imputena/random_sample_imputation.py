import pandas as pd
import numpy as np


def random_sample_imputation(data=None, columns=None, inplace=False):
    """Performs random sample imputation on the data. Missing values in each
    column are replaced by a randomly selected observed values of the same
    column, if available. The operation can be applied to a series, a whole
    dataframe, or a selection of columns of a dataframe.

    :param data: The data on which to perform the random sample imputation
    :type data: pandas.Series or pandas.DataFrame
    :param columns: Columns on which to apply the operation.
    :type columns: array-like, optional
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The series or dataframe with NA values filled in, or
        None if inplace=True.
    :rtype: pandas.Series, pandas.DataFrame, or None
    :raises: TypeError, ValueError
    """
    # Check if data is of the correct type:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    # Treatment if data is a series:
    if isinstance(data, pd.Series):
        if columns is not None:
            raise ValueError('Columns can only be selected if the data is a '
                             'DataFrame.')
        if data.notnull().sum() > 0:
            # The operation is only applied if the column contains some
            # non-NA value.
            number_missing = data.isnull().sum()
            observed_values = data.loc[data.notnull()]
            res.loc[data.isnull()] = np.random.choice(
                observed_values, number_missing, replace=True)
    # Treatment if data is a dataframe:
    if isinstance(data, pd.DataFrame):
        if columns is None:
            columns = data.columns
        for column in columns:
            # Raise error if the column name doesn't exist in the data:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.'
                )
            if data[column].notnull().sum() > 0:
                # The operation is only applied if the column contains some
                # non-NA value.
                number_missing = data[column].isnull().sum()
                observed_values = data.loc[data[column].notnull(), column]
                res.loc[data[column].isnull(), column] = np.random.choice(
                    observed_values, number_missing, replace=True)
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res
