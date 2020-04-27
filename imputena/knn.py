import pandas as pd
import os
import sys
import contextlib

# Workaround to prevent Keras from writing an error message on being imported:
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from fancyimpute import KNN
sys.stderr = stderr


def knn(data=None, columns=None, k=5, inplace=False):
    """Performs k-nearest neighbors imputation on the data. The k nearest
    neighbors or each subject with missing data are chosen and the average
    of their values is used to impute the missing value. The operation can be
    applied to all columns, by leaving the parameter columns empty, or to
    selected columns, passed as an array of strings.

    :param data: The data on which to perform the k-nearest neighbors
        imputation.
    :type data: pandas.DataFrame
    :param columns: Columns on which to apply the operation.
    :type columns: array-like, optional
    :param k: The number of neighbors to which the subject with missing
        values should be compared
    :type k: int, default 5
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The series or dataframe with NA values imputed, or
        None if inplace=True.
    :rtype: pandas.DataFrame or None
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    # Perform KNN
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        knn_out = KNN(k=k).fit_transform(data)
    # Treatment for a whole dataframe:
    if columns is None:
        res[:] = knn_out
    # Treatment for selected columns of a dataframe:
    else:
        for column in columns:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.')
            col_loc = data.columns.get_loc(column)
            res[column] = knn_out[:, col_loc]
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res
