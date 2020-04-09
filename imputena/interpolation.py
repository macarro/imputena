import pandas as pd

def interpolation(data=None, method='linear', columns=None, inplace=False):
    # Check if data is a series or dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # Raise a ValueError if columns are selected for a series:
    if isinstance(data, pd.Series) and columns is not None:
        raise ValueError('Columns can only be selected if the data is a '
                         'DataFrame.')
    if inplace:
        res = data
    else:
        res = data.copy()
    if columns is None:
        if inplace:
            data.interpolate(method=method, inplace=True)
        else:
            res = data.interpolate(method=method, inplace=False)
    else:
        for column in columns:
            # Raise error if the column name doesn't exist in the data:
            if column not in data.columns:
                raise ValueError(
                    '\'' + column + '\' is not a column of the data.')
            res[column] = res[column].interpolate(method=method, inplace=False)
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res
