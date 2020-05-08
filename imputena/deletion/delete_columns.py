import pandas as pd


def delete_columns(data=None, columns=None, threshold=None, inplace=False):
    """Drops variables that contain NA values from the data. If a list of
    column names is passed, all other columns will be ignored, otherwise
    all the columns will be considered. If a threshold is given, the function
    drops those rows which have less non-NA values in the specified columns.

    :param data: The data on which to perform the pairwise dropping of
        variables.
    :type data: pandas.DataFrame
    :param columns: The columns which should be considered. If not passed or
        None, all columns will be considered.
    :type columns: array-like, optional
    :param threshold: Require that many non-NA values in order to not drop a
        column. If not passed or None, all columns with any NA value will be
        dropped.
    :type threshold: int, optional
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The dataframe with columns that contain NA dropped or None if
        inplace=True.
    :rtype: pandas.DataFrame or None
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # If no columns are given, the operation will be applied to all of them:
    if columns is None:
        columns = data.columns
    # Array of columns in which the number of non-NA values is not at least the
    # threshold:
    columns_under_threshold = []
    for column in columns:
        # Raise error if the column name doesn't exist in the data:
        if column not in data.columns:
            raise ValueError(
                '\'' + column + '\' is not a column of the data.'
            )
        if threshold is None:
            # If no threshold is given, drop the column if it contains any NA:
            if data[column].isna().any():
                columns_under_threshold.append(column)
        else:
            # If a threshold is given, drop the column if the number of non-NA
            # values is not at least the threshold:
            if data[column].notna().sum() < threshold:
                columns_under_threshold.append(column)
    # Drop the columns collected in columns_under_threshold:
    if inplace:
        data.drop(columns=columns_under_threshold, inplace=True)
        return None
    else:
        return data.drop(columns=columns_under_threshold)
