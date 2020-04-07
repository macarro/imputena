import pandas as pd
import numpy as np


def random_value_imputation(
        data=None, distribution='uniform', min=0, max=1, sigma=1, mu=0,
        columns=None,
        inplace=False):
    """Fills in missing values with a randomly generated number. If
    distribution is uniform, a float between the min (inclusive) and max (
    exclusive) will be generated. If distribution is normal, a float from a
    normal distribution specified by sigma and int will be generated. If
    distribution is integer, an integer between min (inclusive) and max (
    exclusive) will be drawn from a uniform distribution. If the data is
    passed as a dataframe, the operation can be applied to all columns,
    by leaving the parameter columns empty, or to selected columns, passed
    as an array of strings.

    :param data: The data on which to perform the constant value imputation
    :type data: pandas.Series or pandas.DataFrame
    :param distribution: The distribution from which to draw the random
        values.
    :type distribution: {'uniform', 'normal', 'integer'}
    :param min: The lowest value to be drawn
    :type min: scalar
    :param max: One above the highest value to be drawn
    :type max: scalar
    :param sigma: The sigma value tu be used when drawing from a normal
        distribution.
    :type sigma: scalar
    :param mu: The mu value tu be used when drawing from a normal distribution.
    :type mu: scalar
    :param columns: Columns on which to apply the operation.
    :type columns: array-like, optional
    :param inplace: If True, do operation inplace and return None.
    :type inplace: bool, optional
    :return: The series or dataframe with NA values filled in, or
        None if inplace=True.
    :rtype: pandas.Series, pandas.DataFrame, or None
    :raises: TypeError, ValueError
    """
    # Check if data is a series or dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError('The data has to be a Series or DataFrame.')
    # Raise a ValueError if columns are selected for a series:
    if isinstance(data, pd.Series) and columns is not None:
        raise ValueError('Columns can only be selected if the data is a '
                         'DataFrame.')
    # Check if the distribution has a valid value:
    if distribution not in ['uniform', 'normal', 'integer']:
        raise ValueError(distribution + 'is not a supported distribution')
    # Assign a reference or copy to res, depending on inplace:
    if inplace:
        res = data
    else:
        res = data.copy()
    if isinstance(data, pd.DataFrame):
        if columns is None :
            columns = data.columns
        num_rows = len(res.index)
        num_cols = len(columns)
        if distribution == 'uniform':
            rand = pd.DataFrame(
                (max - min) * np.random.rand(num_rows, num_cols) + min,
                columns=columns,
                index=res.index)
        if distribution == 'normal':
            rand = pd.DataFrame(
                sigma * np.random.randn(num_rows, num_cols) + mu,
                columns=columns,
                index=res.index)
        if distribution == 'integer':
            rand = pd.DataFrame(
                np.random.randint(
                    low=min, high=max, size=(num_rows, num_cols)),
                columns=columns,
                index=res.index)
        res.update(rand, overwrite=False)
    if isinstance(data, pd.Series):
        def get_random_value():
            if distribution == 'uniform':
                return (max - min) * np.random.rand() + min
            if distribution == 'normal':
                return sigma * np.random.randn() + mu
            if distribution == 'integer':
                return np.random.randint(low=min, high=max, size=(1, 1))
        if inplace:
            data.loc[:] = res.apply(
                lambda item: get_random_value() if pd.isnull(item) else item)
        else:
            res = res.apply(
                lambda item: get_random_value() if pd.isnull(item) else item)
    # Return the imputed data, or None if inplace:
    if inplace:
        return None
    else:
        return res
