import pandas as pd


def nocb(data=None, fill_trailing=False, columns=None, inplace=False):
    """Fills in NA values with the next observation in the same column. If
    fill_trailing is true, trailing values are filled in with the last
    observation. The operation can be applied to a series, a whole
    dataframe, or a selection of columns of a dataframe.

    :param data: The data on which to perform the NOCB operation.
    :type data: pandas.Series or pandas.DataFrame
    :param columns: Columns on which to apply the operation.
    :type columns: array-like, optional
    :param fill_trailing: Whether to fill in trailing NA values with the last
        observation.
    :type fill_trailing: bool, default False
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The series or dataframe with NA values filled in, or
        None if inplace=True.
    :rtype: pandas.Series, pandas.DataFrame, or None
    :raises: TypeError, ValueError
    """
    # Check that data is a Series or Dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # Check that each of the given columns is actually a column of data:
    if isinstance(data, pd.Series) and columns is not None:
        raise ValueError('Columns can only be selected if the data is a '
                         'DataFrame.')
    if columns is None:
        # Treatment for a series or all columns of a dataframe
        if inplace:
            # Apply nocb:
            data.fillna(method='bfill', inplace=True)
            # If fill_trailing, apply locf:
            if fill_trailing:
                data.fillna(method='ffill', inplace=True)
            return None
        else:
            if fill_trailing:
                # Apply nocb, then locf:
                return data.fillna(method='bfill').fillna(method='ffill')
            else:
                # Apply nocb only:
                return data.fillna(method='bfill')
    else:
        # Treatment for selected columns of a dataframe
        if inplace:
            res = data
        else:
            res = data.copy()
        for column in columns:
            # Raise error if the column name doesn't exist in the data:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.'
                )
            # Apply nocb:
            res[column].fillna(method='bfill', inplace=True)
            # If fill_trailing, apply locf:
            if fill_trailing:
                res[column].fillna(method='ffill', inplace=True)
        if inplace:
            return None
        else:
            return res
