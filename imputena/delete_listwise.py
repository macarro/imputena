import pandas as pd


def delete_listwise(data=None, threshold=None, inplace=False):
    """Performs listwise deletion on the data.

    :param data: The data on which to perform the listwise deletion of missing
    values.
    :type data: pandas.Series or pandas.DataFrame
    :param threshold: If the data is a DataFrame, require that many non-NA
    values
    :type threshold: int, optional
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, optional
    :return: The series or dataframe with all rows containing NA eliminated, or
    None if inplace=True.
    :rtype: pandas.Series, pandas.DataFrame, or None
    :raises: TypeError, ValueError
    """
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    if isinstance(data, pd.Series) and threshold is not None:
        raise ValueError('A threshold can only be given if the data is a '
                         'DataFrame.')
    kwargs = {}
    if isinstance(data, pd.DataFrame):
        kwargs['thresh'] = threshold
    if inplace:
        data.dropna(inplace=True, **kwargs)
        return None
    else:
        return data.dropna(**kwargs)
