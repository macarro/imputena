import pandas as pd

from .utils import (
    is_series, is_dataframe, is_categorical, is_numeric,
    contains_categorical, contains_only_categorical, is_temporal)


def get_applicable_methods(data=None):
    """Informs about the imputation methods that are applicable to a given
    data frame or series, based on the number of variables (one or
    multiple), type of data (categorical, numerical, o both), and whether
    the data is of temporal nature.

    :param data: The data for which an the applicable imputation method
        should be returned.
    :type data: pandas.Series or pandas.DataFrame
    :return: The imputation methods that are applicable to the data
    :rtype: set of strings
    :raises: TypeError
    """
    # Check that data is a Series or Dataframe:
    if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
        raise TypeError(
            'The data has to be a Series or DataFrame but is a {}.'.format(
                type(data).__name__))
    # Definition of sets:
    applicable_to_cat_only = {
        'logistic regression imputation'
    }
    applicable_to_num_only = {
        'mean substitution',
        'median substitution',
        'random value imputation',
        'linear regression',
        'stochastic regression',
        'imputation using k-NN',
        'interpolation',
        'interpolation with seasonal adjustment'
    }
    applicable_to_cat_and_num = {
        'listwise deletion',
        'pairwise deletion',
        'variable deletion',
        'random sample imputation',
        'random hot-deck imputation',
        'most-frequent substitution',
        'constant value substitution',
        'srmi',
        'mice',
        'LOCF',
        'NOCB'
    }
    applicable_to_cat = applicable_to_cat_only.union(applicable_to_cat_and_num)
    applicable_to_num = applicable_to_num_only.union(applicable_to_cat_and_num)
    all_methods = applicable_to_cat.union(applicable_to_num)
    requires_temp = {
        'LOCF',
        'NOCB',
        'interpolation',
        'interpolation with seasonal adjustment'
    }
    does_not_require_temp = all_methods - requires_temp
    applicable_to_series = {
        'listwise deletion',
        'random sample imputation',
        'most-frequent substitution',
        'constant value substitution',
        'LOCF',
        'NOCB',
        'mean substitution',
        'median substitution',
        'random value imputation',
        'interpolation',
        'interpolation with seasonal adjustment'
    }
    # Applicability:
    res = set()
    if is_series(data):
        if is_categorical(data):
            if is_temporal(data):
                res = applicable_to_series.intersection(
                    applicable_to_cat
                )
            else:
                res = applicable_to_series.intersection(
                    applicable_to_cat.intersection(does_not_require_temp)
                )
        else:
            if is_temporal(data):
                res = applicable_to_series.intersection(
                    applicable_to_num
                )
            else:
                res = applicable_to_series.intersection(
                    applicable_to_num.intersection(does_not_require_temp)
                )
    if is_dataframe(data):
        if contains_only_categorical(data):
            if is_temporal(data):
                res = applicable_to_cat
            else:
                res = applicable_to_cat.intersection(does_not_require_temp)
        elif contains_categorical(data):
            if is_temporal(data):
                res = applicable_to_cat_and_num
            else:
                res = applicable_to_cat_and_num.intersection(
                    does_not_require_temp)
        else:
            if is_temporal(data):
                res = applicable_to_num
            else:
                res = applicable_to_num.intersection(does_not_require_temp)
    # Return set with applicable methods:
    return res
