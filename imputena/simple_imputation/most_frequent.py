import pandas as pd


def most_frequent(data=None, columns=None, inplace=False):
    """Fills in missing values with the most frequent value (mode) in the
    same column, in case of a dataframe, or in the series as a whole in case
    of a series. If the data is passed as a dataframe, the operation can be
    applied to all columns, by leaving the parameter columns empty; or to
    selected columns, passed as an array of strings.

    :param data: The data on which to perform the most frequent imputation.
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
    # Check if data is a series or dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # Raise a ValueError if columns are selected for a series:
    if isinstance(data, pd.Series) and columns is not None:
        raise ValueError('Columns can only be selected if the data is a '
                         'DataFrame.')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    if columns is None:
        # Treatment for a series or all columns of a dataframe
        res.fillna(data.mode().iloc[0], inplace=True)
    else:
        # Treatment for selected columns of a dataframe
        for column in columns:
            # Raise error if the column name doesn't exist in the data:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.'
                )
            # Impute the missing values of the column
            res[column].fillna(data[column].mode().iloc[0], inplace=True)
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res
