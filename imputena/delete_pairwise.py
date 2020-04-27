import pandas as pd


def delete_pairwise(data=None, columns=None, threshold=None, inplace=False):
    """Performs pairwise deletion on the data: Drops any rows that contain NA
    values in any of the specified columns. If a threshold is given, the
    function drops those rows which have less non-NA values in the specified
    columns.

    :param data: The data on which to perform the pairwise deletion of missing
        values.
    :type data: pandas.DataFrame
    :param columns: rows will be dropped if any of their value in any of those
        columns is NA.
    :type columns: array-like
    :param threshold: Require that many non-NA values in the specified columns
    :type threshold: int, optional
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The dataframe with all rows containing NA in one or more of the
        specified columns eliminated or None if inplace=True.
    :rtype: pandas.DataFrame or None
    :raises: TypeError, ValueError
    """
    # Check that data is a Dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Check that each of the given columns is actually a column of data:
    for column in columns:
        if column not in data.columns:
            raise ValueError('\'' + column + '\' is not a column of the data.')
    # Apply the pairwise deletion and return the result, or None if inplace:
    if inplace:
        data.dropna(inplace=True, thresh=threshold, subset=columns)
        return None
    else:
        return data.dropna(thresh=threshold, subset=columns)
