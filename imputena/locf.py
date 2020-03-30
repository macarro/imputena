import pandas as pd


def locf(data=None, columns=None, fill_leading=False, inplace=False):
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    if isinstance(data, pd.Series) and columns is not None:
        raise ValueError('Columns can only be selected if the data is a '
                         'DataFrame.')
    if columns is None:
        # Treatment for a series or all columns of a dataframe:
        if inplace:
            data.fillna(method='ffill', inplace=True)
            if fill_leading:
                data.fillna(method='bfill', inplace=True)
            return None
        else:
            if fill_leading:
                return data.fillna(method='ffill').fillna(method='bfill')
            else:
                return data.fillna(method='ffill')
    else:
        if inplace:
            for column in columns:
                # Raise error if the column name doesn't exist in the data:
                if column not in data.columns:
                    raise ValueError(
                        '\'' + column + '\' is not a column of the data.'
                    )
                # Apply locf:
                data[column].fillna(method='ffill', inplace=True)
                # If fill_leading, apply nocb:
                if fill_leading:
                    data[column].fillna(method='bfill', inplace=True)
            return None
        else:
            res = data.copy()
            for column in columns:
                # Raise error if the column name doesn't exist in the data:
                if column not in data.columns:
                    raise ValueError(
                        '\'' + column + '\' is not a column of the data.'
                    )
                # Apply locf:
                res[column].fillna(method='ffill', inplace=True)
                # If fill_leading, apply nocb:
                if fill_leading:
                    res[column].fillna(method='bfill', inplace=True)
            return res

