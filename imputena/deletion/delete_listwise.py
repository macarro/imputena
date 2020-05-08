import pandas as pd


def delete_listwise(data=None, threshold=None, inplace=False):
    """Performs listwise deletion on the data: Drops any rows that contain
    missing values. If a threshold is given, the function drops those rows
    which have less non-NA values. If the operation should only affect certain
    columns, user delete_pairwise instead.

    :param data: The data on which to perform the listwise deletion of missing
        values.
    :type data: pandas.Series or pandas.DataFrame
    :param threshold: If the data is a DataFrame, require that many non-NA
        values
    :type threshold: int, optional
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The series or dataframe with all rows containing NA eliminated, or
        None if inplace=True.
    :rtype: pandas.Series, pandas.DataFrame, or None
    :raises: TypeError, ValueError
    """
    # Check that data is a Series or Dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # If a threshold is given while data is a Series, raise a ValueError:
    if isinstance(data, pd.Series) and threshold is not None:
        raise ValueError('A threshold can only be given if the data is a '
                         'DataFrame.')
    # kwargs for dropna:
    kwargs = {}
    # Add threshold to dropna kwargs:
    if isinstance(data, pd.DataFrame):
        kwargs['thresh'] = threshold
    # Apply the listwise deletion and return the result, or None if inplace:
    if inplace:
        data.dropna(inplace=True, **kwargs)
        return None
    else:
        return data.dropna(**kwargs)
