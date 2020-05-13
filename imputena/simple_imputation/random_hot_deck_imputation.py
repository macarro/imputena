import pandas as pd
import numpy as np


def random_hot_deck_imputation(
        data=None, incomplete_variable=None, deck_variables=None,
        inplace=False):
    """Performs random hot deck imputation on the data. Missing values receive
    a valid value from a donor randomly chosen from a pool. The pool is
    different for each row containing a missing value in incomplete_variable
    and consists of all rows which coincide in value with the incomplete row
    for all of the columns in deck_variables.

    :param data: The data on which to perform the random hot deck imputation.
    :type data: pandas.DataFrame
    :param incomplete_variable: The variable in which the missing values
        should be imputed.
    :type incomplete_variable: String
    :param deck_variables: The donor has to have the same value as the row
        for these variables.
    :type deck_variables: array-like
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, default False
    :return: The dataframe with random hot deck imputation performed for the
        incomplete variable or None if inplace=True.
    :rtype: pandas.DataFrame or None
    :raises: TypeError, ValueError
    """
    # Check if data is a dataframe:
    if not isinstance(data, pd.DataFrame):
        raise TypeError('The data has to be a DataFrame.')
    # Check if the incomplete variable is actually a column of the dataframe:
    if incomplete_variable not in data.columns:
        raise ValueError(
            '\'' + incomplete_variable + '\' is not a column of the data.')
    # Check if each of the deck variables is actually a column of the
    # dataframe:
    for column in deck_variables:
        if column not in data.columns:
            raise ValueError('\'' + column + '\' is not a column of the data.')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    # Implementation using a apply:
    res.loc[:, :] = data.apply(
        lambda row: get_row_with_donation(
            row, incomplete_variable, deck_variables, data),
        axis=1, result_type='broadcast')
    # Return dataframe is the operation is not to be performed inplace:
    if not inplace:
        return res


def get_row_with_donation(row, incomplete_variable, deck_variables, data):
    """This function imputes the value of the incomplete variable in the
    passed row using random hot deck imputation and returns the whole row so
    that the function can be applied to each row of a dataframe.

    :param row: The row for which the missing value should be imputed
    :type row: pandas.Series
    :param incomplete_variable: The variable for which the row might contain a
        missing value
    :type incomplete_variable: String
    :param deck_variables: The donor has to have the same value as the row
        for these variables.
    :type deck_variables: array-like
    :param data: The whole dataframe, used to select the donor
    :type data: pandas.DataFrame
    :return: The row with the missing value imputed
    :rtype: pandas.Series
    """

    res = row.copy()
    # Since this function is applied to each row of the database,
    # it is necessary to check whether it actually contains a missing value
    # for the incomplete variable:
    if pd.isna(row[incomplete_variable]):
        # query_string will hold a query selecting all possible donors.
        # First, the donor should not have a missing value in the variable
        # that is to be imputed:
        query_string = '`' + incomplete_variable + '`.notnull()'
        # The donor should also coincide in value with the row that the
        # operation if being performed on for all the deck variables:
        for variable in deck_variables:
            query_string += ' and (`' + variable + '` == \'' \
                            + str(row[variable]) + '\')'
        donors = data.query(query_string)
        # The missing value is only imputed if a donor that coincides in
        # value for all the deck variables exists:
        if len(donors) > 0:
            value = donors.iloc[np.random.randint(0, len(donors))][
                data.columns.get_loc(incomplete_variable)]
            res = row.copy()
            res[incomplete_variable] = value
    return res
