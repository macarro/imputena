import pandas as pd
import numpy as np


def hot_deck_imputation(
        data=None, incomplete_variable=None, deck_variables=None,
        inplace=False):
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Implementation using a apply:
    if inplace:
        data.loc[:, :] = data.apply(
            lambda row: get_row_with_donation(
                row, incomplete_variable, deck_variables, data),
            axis=1, result_type='broadcast')
    else:
        res = data.apply(
            lambda row: get_row_with_donation(
                row, incomplete_variable, deck_variables, data),
            axis=1, result_type='broadcast')
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res


def get_row_with_donation(row, incomplete_variable, deck_variables, data):
    res = row.copy()

    if np.isnan(row[incomplete_variable]):
        query_string = incomplete_variable + '.notnull()'
        for variable in deck_variables:
            query_string += ' and (' + variable + ' == \'' \
                            + str(row[variable]) + '\')'

        donors = data.query(query_string)
        if len(donors) > 0:
            value = donors.iloc[np.random.randint(0, len(donors))][
                data.columns.get_loc(incomplete_variable)]
            res = row.copy()
            res[incomplete_variable] = value
    return res
