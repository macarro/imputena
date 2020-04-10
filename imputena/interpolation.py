import pandas as pd


def interpolation(
        data=None, method='linear', direction='both', columns=None,
        inplace=False):
    """Performs linear, quadratic, or cubic interpolation on a series or a
    data frame. If the data is passed as a dataframe, the operation can be
    applied to all columns, by leaving the parameter columns empty, or to
    selected columns, passed as an array of strings.

    :param data: The data on which to perform the interpolation.
    :type data: pandas.Series or pandas.DataFrame
    :param method: The interpolation model to use.
    :type method: {'linear', 'quadratic', 'cubic'}, default 'linear'
    :param direction: Direction in which to interpolate values when the
        interpolation method is linear.
    :type direction: {'forward', 'backward', 'both'}, default 'both'
    :param columns: Columns on which to apply the operation.
    :type columns: array-like, optional
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The series or dataframe with NA values interpolated, or
        None if inplace=True.
    :rtype: pandas.Series, pandas.DataFrame, or None
    :raises: TypeError, ValueError
    """
    # Check that data is a series or dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # Raise a ValueError if columns are selected for a Series:
    if isinstance(data, pd.Series) and columns is not None:
        raise ValueError('Columns can only be selected if the data is a '
                         'DataFrame.')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    # kwargs for the pandas interpolate() function:
    int_kwargs = {'limit_direction': direction}
    # Treatment for a whole DataFrame or a Series:
    if columns is None:
        if inplace:
            data.interpolate(method=method, inplace=True, **int_kwargs)
        else:
            res = data.interpolate(method=method, inplace=False, **int_kwargs)
    # Treatment for selected columns:
    else:
        for column in columns:
            # Raise error if the column name doesn't exist in the data:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.')
            res[column] = res[column].interpolate(
                method=method, inplace=False, **int_kwargs)
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res
