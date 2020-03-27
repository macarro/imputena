import pandas as pd


def delete_listwise(data=None, inplace=False):
    """Performs listwise deletion on the data.

    :param data: The data on which to perform the listwise deletion of missing
    values.
    :type data: pandas.Series or pandas.DataFrame
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, optional
    :return: The series or dataframe with all rows containing NA eliminated.
    :rtype: pandas.Series or pandas.DataFrame, depending on the input
    :raises: TypeError
    """
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    if inplace:
        data.dropna(inplace=True)
        return None
    else:
        return data.dropna()
